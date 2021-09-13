import pandas as pd
import numpy as np
import math
import UNI_v3_funcs
import copy

class StrategyObservation:
    def __init__(self,timepoint,current_price,strategy_in,liquidity_in_0,liquidity_in_1,fee_tier,
                 decimals_0,decimals_1,token_0_left_over=0.0,token_1_left_over=0.0,
                 token_0_fees=0.0,token_1_fees=0.0,liquidity_ranges=None,strategy_info = None,swaps=None):
        
        ######################################
        # 1. Store current values
        ######################################
        
        self.time                  = timepoint
        self.price                 = current_price
        self.liquidity_in_0        = liquidity_in_0
        self.liquidity_in_1        = liquidity_in_1
        self.fee_tier              = fee_tier
        self.decimals_0            = decimals_0
        self.decimals_1            = decimals_1
        self.token_0_left_over     = token_0_left_over
        self.token_1_left_over     = token_1_left_over
        self.token_0_fees_accum    = token_0_fees
        self.token_1_fees_accum    = token_1_fees
        self.reset_point           = False
        self.reset_reason          = ''
        self.decimal_adjustment    = 10**(self.decimals_1  - self.decimals_0)
        self.tickSpacing           = int(self.fee_tier*2*10000)   
        self.token_0_fees          = 0.0
        self.token_1_fees          = 0.0
        
        
        TICK_P_PRE                 = int(math.log(self.decimal_adjustment*self.price,1.0001))        
        self.price_tick            = round(TICK_P_PRE/self.tickSpacing)*self.tickSpacing
            
        ######################################
        # 2. Execute the strategy
        #    If this is the first observation, need to generate ranges 
        #    Otherwise, check if a rebalance is required and execute.
        #    If swaps data has been fed in, it will be used to estimate fee income (for backtesting simulations)
        #    Otherwise just the ranges will be updated (for a live environment)
        ######################################
        if liquidity_ranges is None and strategy_info is None:
            self.liquidity_ranges,self.strategy_info  = strategy_in.set_liquidity_ranges(self)
                                 
        else: 
            self.liquidity_ranges         = copy.deepcopy(liquidity_ranges)
            
            # Update amounts in each position according to current pool price
            for i in range(len(self.liquidity_ranges)):
                self.liquidity_ranges[i]['time'] = self.time                
                amount_0, amount_1 = UNI_v3_funcs.get_amounts(self.price_tick,
                                                             self.liquidity_ranges[i]['lower_bin_tick'],
                                                             self.liquidity_ranges[i]['upper_bin_tick'],
                                                             self.liquidity_ranges[i]['position_liquidity'],
                                                             self.decimals_0,
                                                             self.decimals_1)

                self.liquidity_ranges[i]['token_0'] = amount_0
                self.liquidity_ranges[i]['token_1'] = amount_1
                
                if swaps is not None:
                    fees_token_0,fees_token_1           = self.accrue_fees(swaps)
                    self.token_0_fees                   = fees_token_0
                    self.token_1_fees                   = fees_token_1
                
            self.liquidity_ranges,self.strategy_info     = strategy_in.check_strategy(self,strategy_info)
                
    ########################################################
    # Accrue earned fees (not supply into LP yet)
    ########################################################               
    def accrue_fees(self,relevant_swaps):   
        
        fees_earned_token_0 = 0.0
        fees_earned_token_1 = 0.0
                
        if len(relevant_swaps) > 0:
            # For every swap in this time period
            for s in range(len(relevant_swaps)):
                for i in range(len(self.liquidity_ranges)):
                    in_range   = (self.liquidity_ranges[i]['lower_bin_tick'] <= relevant_swaps.iloc[s]['tick_swap']) and \
                                 (self.liquidity_ranges[i]['upper_bin_tick'] >= relevant_swaps.iloc[s]['tick_swap'])

                    token_0_in = relevant_swaps.iloc[s]['token_in'] == 'token0'
                    fraction_fees_earned_position = self.liquidity_ranges[i]['position_liquidity']/relevant_swaps.iloc[s]['virtual_liquidity']

                    fees_earned_token_0 += in_range * token_0_in     * self.fee_tier * fraction_fees_earned_position * relevant_swaps.iloc[s]['traded_in']
                    fees_earned_token_1 += in_range * (1-token_0_in) * self.fee_tier * fraction_fees_earned_position * relevant_swaps.iloc[s]['traded_in']
        
        self.token_0_fees_accum += fees_earned_token_0
        self.token_1_fees_accum += fees_earned_token_1
        
        return fees_earned_token_0,fees_earned_token_1            
     
    ########################################################
    # Rebalance: Remove all liquidity positions
    # Not dependent on strategy
    ########################################################   
    def remove_liquidity(self):
    
        removed_amount_0    = 0.0
        removed_amount_1    = 0.0
        
        # For every bin, get the amounts you currently have and withdraw
        for i in range(len(self.liquidity_ranges)):
            
            position_liquidity = self.liquidity_ranges[i]['position_liquidity']
           
            TICK_A             = self.liquidity_ranges[i]['lower_bin_tick']
            TICK_B             = self.liquidity_ranges[i]['upper_bin_tick']
            
            token_amounts      = UNI_v3_funcs.get_amounts(self.price_tick,TICK_A,TICK_B,
                                                     position_liquidity,self.decimals_0,self.decimals_1)   
            removed_amount_0   += token_amounts[0]
            removed_amount_1   += token_amounts[1]
        
        self.liquidity_in_0 = removed_amount_0 + self.token_0_left_over + self.token_0_fees_accum
        self.liquidity_in_1 = removed_amount_1 + self.token_1_left_over + self.token_1_fees_accum
        
        self.token_0_left_over = 0.0
        self.token_1_left_over = 0.0
        
        self.token_0_fees_accum = 0.0
        self.token_1_fees_accum = 0.0
        
   
########################################################
# Simulate reset strategy using a Pandas series called price_data, which has as an index
# the time point, and contains the pool price (token 1 per token 0)
########################################################

def simulate_strategy(price_data,swap_data,strategy_in,
                       liquidity_in_0,liquidity_in_1,fee_tier,decimals_0,decimals_1):

    strategy_results = []    
  
    # Go through every time period in the data that was passet
    for i in range(len(price_data)): 
        # Strategy Initialization
        if i == 0:
            strategy_results.append(StrategyObservation(price_data.index[i],
                                              price_data[i],
                                              strategy_in,
                                              liquidity_in_0,liquidity_in_1,
                                              fee_tier,decimals_0,decimals_1))
        # After initialization
        else:
            
            relevant_swaps = swap_data[price_data.index[i-1]:price_data.index[i]]
            strategy_results.append(StrategyObservation(price_data.index[i],
                                              price_data[i],
                                              strategy_in,
                                              strategy_results[i-1].liquidity_in_0,
                                              strategy_results[i-1].liquidity_in_1,
                                              strategy_results[i-1].fee_tier,
                                              strategy_results[i-1].decimals_0,
                                              strategy_results[i-1].decimals_1,
                                              strategy_results[i-1].token_0_left_over,
                                              strategy_results[i-1].token_1_left_over,
                                              strategy_results[i-1].token_0_fees,
                                              strategy_results[i-1].token_1_fees,
                                              strategy_results[i-1].liquidity_ranges,
                                              strategy_results[i-1].strategy_info,
                                              relevant_swaps
                                              ))
                
    return strategy_results

########################################################
# Extract Strategy Data
########################################################

def generate_simulation_series(simulations,strategy_in):
    data_strategy                    = pd.DataFrame([strategy_in.dict_components(i) for i in simulations])
    data_strategy                    = data_strategy.set_index('time',drop=False)
    data_strategy                    = data_strategy.sort_index()
    return data_strategy


########################################################
# Calculates % returns over a minutes frequency
########################################################

def aggregate_time(data,minutes = 10):
    price_range               = pd.DataFrame({'time_pd': pd.date_range(data.index.min(),data.index.max(),freq='1 min',tz='UTC')})
    price_range               = price_range.set_index('time_pd',drop=False)
    new_data                  = price_range.merge(data,left_index=True,right_index=True,how='left')
    new_data['baseCurrency']  = new_data['baseCurrency'].ffill()
    new_data['quoteCurrency'] = new_data['quoteCurrency'].ffill()
    new_data['baseAmount']    = new_data['baseAmount'].ffill()
    new_data['quoteAmount']   = new_data['quoteAmount'].ffill()
    new_data['quotePrice']    = new_data['quotePrice'].ffill()
    price_set                 = set(pd.date_range(new_data.index.min(),new_data.index.max(),freq=str(minutes)+'min'))
    return new_data[new_data.index.isin(price_set)]

def aggregate_price_data(data,minutes,PRICE_CHANGE_LIMIT = .9):
    price_data_aggregated                 = aggregate_time(data,minutes).copy()
    price_data_aggregated['price_return'] = (price_data_aggregated['quotePrice'].pct_change())
    price_data_aggregated['log_return']   = np.log1p(price_data_aggregated.price_return)
    price_data_full                       = price_data_aggregated[1:]
    price_data_filtered                   = price_data_full[(price_data_full['price_return'] <= PRICE_CHANGE_LIMIT) & (price_data_full['price_return'] >= -PRICE_CHANGE_LIMIT) ]
    return price_data_filtered

def analyze_strategy(data_in,initial_position_value,token_0_usd_data=None):

    # For pools where token0 is a USD stable coin, no need to supply token_0_usd
    # Otherwise must pass the USD price data for token 0
    
    if token_0_usd_data is None:
        data_usd = data_in
        data_usd['cum_fees_usd']       = data_usd['token_0_fees'].cumsum() + (data_usd['token_1_fees'] * data_usd['price_1_0']).cumsum()
        data_usd['value_position_usd'] = data_usd['value_position']
    else:
        # Merge in usd price data
        token_0_usd_data['price_0_usd'] = 1/token_0_usd_data['quotePrice']
        token_0_usd_data                = token_0_usd_data.sort_index()
        data_in['time_pd']              = pd.to_datetime(data_in['time'],utc=True)
        data_in                         = data_in.set_index('time_pd')
        data_usd                        = pd.merge_asof(data_in,token_0_usd_data['price_0_usd'],on='time_pd',direction='backward',allow_exact_matches = True)
        
        # Compute accumulated fees and other usd metrics
        data_usd['cum_fees_0']          = data_usd['token_0_fees'].cumsum() + (data_usd['token_1_fees'] * data_usd['price_1_0']).cumsum()
        data_usd['cum_fees_usd']        = data_usd['cum_fees_0']*data_usd['price_0_usd']
        data_usd['value_position_usd']  = data_usd['value_position']*data_usd['price_0_usd']


    days_strategy           = (data_usd['time'].max()-data_usd['time'].min()).days    
    strategy_last_obs       = data_usd.tail(1)
    strategy_last_obs       = strategy_last_obs.reset_index(drop=True)
    net_apr                 = float((strategy_last_obs['value_position_usd']/initial_position_value - 1) * 365 / days_strategy)

    summary_strat = {
                        'days_strategy'        : days_strategy,
                        'gross_fee_apr'        : float((strategy_last_obs['cum_fees_usd']/initial_position_value) * 365 / days_strategy),
                        'gross_fee_return'     : float(strategy_last_obs['cum_fees_usd']/initial_position_value),
                        'net_apr'              : net_apr,
                        'net_return'           : float(strategy_last_obs['value_position_usd']/initial_position_value  - 1),
                        'rebalances'           : data_usd['reset_point'].sum(),
                        'max_drawdown'         : ( data_usd['value_position_usd'].max() - data_usd['value_position_usd'].min() ) / data_usd['value_position_usd'].max(),
                        'volatility'           : ((data_usd['value_position_usd'].pct_change().var())**(0.5)) * ((365*24*60)**(0.5)), # Minute frequency data
                        'sharpe_ratio'         : float(net_apr / (((data_usd['value_position_usd'].pct_change().var())**(0.5)) * ((365*24*60)**(0.5)))),
                        'mean_base_position'   : (data_usd['base_position_value']/ \
                                                  (data_usd['base_position_value']+data_usd['limit_position_value']+data_usd['value_left_over'])).mean(),
                        'median_base_position' : (data_usd['base_position_value']/ \
                                                  (data_usd['base_position_value']+data_usd['limit_position_value']+data_usd['value_left_over'])).median()
                    }
    
    return summary_strat


def plot_strategy(data_strategy,y_axis_label,base_color = '#ff0000'):
    import plotly.graph_objects as go
    
    CHART_SIZE = 300

    fig_strategy = go.Figure()
    fig_strategy.add_trace(go.Scatter(
        x=data_strategy['time'], 
        y=1/data_strategy['base_range_lower'],
        fill=None,
        mode='lines',
        showlegend = False,
        line_color=base_color,
        ))
    fig_strategy.add_trace(go.Scatter(
        x=data_strategy['time'], 
        y=1/data_strategy['base_range_upper'],
        name='Base Position',
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', line_color=base_color))

    fig_strategy.add_trace(go.Scatter(
        x=data_strategy['time'], 
        y=1/data_strategy['limit_range_lower'],
        fill=None,
        mode='lines',
        showlegend = False,
        line_color='#6f6f6f'))

    fig_strategy.add_trace(go.Scatter(
        x=data_strategy['time'], 
        y=1/data_strategy['limit_range_upper'],
        name='Base + Limit Position',
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', line_color='#6f6f6f',))

    fig_strategy.add_trace(go.Scatter(
        x=data_strategy['time'], 
        y=1/data_strategy['reset_range_lower'],
        name='Strategy Reset Bound',
        line=dict(width=2,dash='dot',color='black')))

    fig_strategy.add_trace(go.Scatter(
        x=data_strategy['time'], 
        y=1/data_strategy['reset_range_upper'],
        showlegend = False,
        line=dict(width=2,dash='dot',color='black',)))

    fig_strategy.add_trace(go.Scatter(
        x=data_strategy['time'], 
        y=data_strategy['price_1_0'],
        name='Price',
        line=dict(width=2,color='black')))

    fig_strategy.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height= CHART_SIZE,
        title = 'Autoregressive Strategy Simulation',
        xaxis_title="Date",
        yaxis_title=y_axis_label,
    )

    fig_strategy.show(renderer="png")