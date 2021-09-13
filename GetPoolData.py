import pandas as pd
from datetime import datetime, timedelta
import requests
import pickle
import importlib
from itertools import compress
    
# Extract all Mint, Burn, and Swap Events
# From a given pool
# Returns json requests

def query_univ3_graph(query: str, variables=None) -> dict:
    univ3_graph_url = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'
    """Make graphql query to subgraph"""
    if variables:
        params = {'query': query, 'variables': variables}
    else:
        params = {'query': query}
    response = requests.post(univ3_graph_url, json=params)
    return response.json()

def get_swap_data(contract_address,file_name,DOWNLOAD_DATA=False):        
        
    request_swap = [] 
    
    if DOWNLOAD_DATA:

        current_payload = generate_fist_event_payload('swaps',contract_address)
        current_id      = query_univ3_graph(current_payload)['data']['pool']['swaps'][0]['id']
        finished        = False

        while not finished:
            current_payload = generate_event_payload('swaps',contract_address,str(1000))
            response        = query_univ3_graph(current_payload,{'paginateId':current_id})['data']['pool']['swaps']

            if len(response) == 0:
                finished = True
            else:
                current_id = response[-1]['id']
                request_swap.extend(response)
                
            with open('./data/'+file_name+'_swap.pkl', 'wb') as output:
                pickle.dump(request_swap, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open('./data/'+file_name+'_swap.pkl', 'rb') as input:
            request_swap = pickle.load(input)
           
    return pd.DataFrame(request_swap)

##############################################################
# Get Pool Virtual Liquidity Data using Flipside Data Pool Stats Table
##############################################################
def get_liquidity_flipside(flipside_query,file_name,DOWNLOAD_DATA = False):
    

    if DOWNLOAD_DATA:        
        for i in flipside_query:
            request_stats    = [pd.DataFrame(requests.get(x).json()) for x in flipside_query]
        with open('./data/'+file_name+'_liquidity.pkl', 'wb') as output:
            pickle.dump(request_stats, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open('./data/'+file_name+'_liquidity.pkl', 'rb') as input:
            request_stats = pickle.load(input)            
            
    stats_data                      = pd.concat(request_stats)
    
   
    return stats_data
    
##############################################################
# Get all swaps for the pool using flipside data's price feed
# For the contract's liquidity
##############################################################
def get_pool_data_flipside(contract_address,flipside_query,file_name,DOWNLOAD_DATA = False):

    # Download  events
    swap_data               = get_swap_data(contract_address,file_name,DOWNLOAD_DATA)
    swap_data['time_pd']    = pd.to_datetime(swap_data['timestamp'], unit='s', origin='unix',utc=True)
    swap_data               = swap_data.set_index('time_pd')
    swap_data['tick_swap']  = swap_data['tick']
    swap_data               = swap_data.sort_index()
    
    # Download pool liquidity data
    stats_data              = get_liquidity_flipside(flipside_query,file_name,DOWNLOAD_DATA)    
    stats_data['time_pd']   = pd.to_datetime(stats_data['BLOCK_TIMESTAMP'], origin='unix',utc=True) 
    stats_data              = stats_data.set_index('time_pd')
    stats_data              = stats_data.sort_index()
    stats_data['tick_pool'] = stats_data['TICK']
    
    full_data               = pd.merge_asof(swap_data,stats_data[['VIRTUAL_LIQUIDITY_ADJUSTED','tick_pool']],on='time_pd',direction='backward',allow_exact_matches = False)
    full_data               = full_data.set_index('time_pd')
    # token with negative amounts is the token being swapped in
    full_data['tick_swap']       = full_data['tick_swap'].astype(int)
    full_data['amount0']         = full_data['amount0'].astype(float)
    full_data['amount1']         = full_data['amount1'].astype(float)
    full_data['token_in']        = full_data.apply(lambda x: 'token0' if (x['amount0'] < 0) else 'token1',axis=1)
    
    return full_data

##############################################################
# Get Price Data from Bitquery
##############################################################
def get_price_data_bitquery(token_0_address,token_1_address,date_begin,date_end,api_token,file_name,DOWNLOAD_DATA = False,RATE_LIMIT=True):

    request = []
    
    if DOWNLOAD_DATA:        
        if RATE_LIMIT:
            # Break out into months to rate limit
            months_to_request = pd.date_range(date_begin,date_end,freq="M").strftime("%Y-%m-%d").tolist()
                
            for i in range(len(months_to_request)-1):             
                request.append(run_query(generate_price_payload(token_0_address,token_1_address,months_to_request[i],months_to_request[i+1]),api_token))
            with open('./data/'+file_name+'_1min.pkl', 'wb') as output:
                pickle.dump(request, output, pickle.HIGHEST_PROTOCOL)
        else:
            # Otherwise just download the data
            request.append(run_query(generate_price_payload(token_0_address,token_1_address,date_begin,date_end),api_token))
    else:
        with open('./data/'+file_name+'_1min.pkl', 'rb') as input:
            request = pickle.load(input)

    # Prepare data for strategy:
    # Collect json data and add to a pandas Data Frame
    
    requests_with_data = [len(x['data']['ethereum']['dexTrades']) > 0 for x in request]
    relevant_requests  = list(compress(request, requests_with_data))
    
    price_data = pd.concat([pd.DataFrame({
    'time':           [x['timeInterval']['minute'] for x in request_price['data']['ethereum']['dexTrades']],
    'baseCurrency':   [x['baseCurrency']['symbol'] for x in request_price['data']['ethereum']['dexTrades']],
    'quoteCurrency':  [x['quoteCurrency']['symbol'] for x in request_price['data']['ethereum']['dexTrades']],
    'quoteAmount':    [x['quoteAmount'] for x in request_price['data']['ethereum']['dexTrades']],
    'baseAmount':     [x['baseAmount'] for x in request_price['data']['ethereum']['dexTrades']],
    'quotePrice':     [x['quotePrice'] for x in request_price['data']['ethereum']['dexTrades']]
    }) for request_price in relevant_requests])
    
    price_data['time']    = pd.to_datetime(price_data['time'], format = '%Y-%m-%d %H:%M:%S')
    price_data['time_pd'] = pd.to_datetime(price_data['time'],utc=True)
    price_data            = price_data.set_index('time_pd')

    return price_data

##############################################################
# Generate payload for bitquery events
##############################################################


def generate_event_payload(event,address,n_query):
        payload =   '''
            query($paginateId: String!){
              pool(id:"'''+address+'''"){
                '''+event+'''(
                  first: '''+n_query+'''
                  orderBy: id
                  orderDirection: asc
                  where: {
                    id_gt: $paginateId
                  }
                ) {
                  id
                  timestamp
                  tick
                  amount0
                  amount1
                  amountUSD
                }
              }
            }'''
        return payload
    
def generate_fist_event_payload(event,address):
        payload = '''query{
                      pool(id:"'''+address+'''"){
                      '''+event+'''(
                      first: 1
                      orderBy: id
                      orderDirection: asc
                        ) {
                          id
                          timestamp
                          tick
                          amount0
                          amount1
                        }
                      }
                    }'''
        return payload

def generate_price_payload(token_0_address,token_1_address,date_begin,date_end):
    payload =   '''{
                  ethereum(network: ethereum) {
                    dexTrades(
                      options: {asc: "timeInterval.minute"}
                      date: {between: ["'''+date_begin+'''","'''+date_end+'''"]}
                      exchangeName: {is: "Uniswap"}
                      baseCurrency: {is: "'''+token_0_address+'''"}
                      quoteCurrency: {is: "'''+token_1_address+'''"}

                    ) {
                      timeInterval {
                        minute(count: 1)
                      }
                      baseCurrency {
                        symbol
                        address
                      }
                      baseAmount
                      quoteCurrency {
                        symbol
                        address
                      }
                      quoteAmount
                      quotePrice
                    }
                  }
                }'''
    
    return payload

# Make dependent on smart contract?
#smartContractAddress: {is: "'''+contract_address+'''"}   
##############################################################
# A simple function to use requests.post to make the API call
##############################################################
def run_query(query,api_token):  
    url       = 'https://graphql.bitquery.io/'
    headers = {'X-API-KEY': api_token}
    request = requests.post(url,
                            json={'query': query}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception('Query failed and return code is {}.      {}'.format(request.status_code,query))