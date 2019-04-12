#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:02:22 2019

@author: roshanprakash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Trader:
    def __init__(self, **kwargs):
        """
        A class that helps formulate the trading decision-making problem with
        a Discrete Time Markov Chain.
        
        PARAMETERS
        ----------
        A dictionary with keys indicating the name of the stock (str) and value 
        indicating the path name to the csv file that contains historical data
        for the stock (str).
        
        RETURNS
        -------
        None
        """
        self.train_data = {}
        self.test_data = {}
        self.trigger_states = {}
        self.state_ranges = {}
        self.transition_matrix = {}
        self.portfolio = {'balance':2500000.0}
        self.transaction_volume=1000
        self.commission = 0.015
        for key, path in kwargs.items():
            try:
                df = pd.read_csv(path, usecols=['Date', 'Close', 'Open']).sort_values('Date')
            except:
                df = pd.read_csv(path, usecols=['date', 'close', 'open']).sort_values('date')
            df['delta'] = df.Close.pct_change()
            df['EMA'] = self.compute_EMA(df.Close)
            df['LT_MA'] = self.compute_MA(df.Close,long_term=True)
            df['ST_MA'] = self.compute_MA(df.Close,long_term=False)
            df['MACD'], df['signal_line'] = self.compute_momentum_signals(df.Close)
            df['ST_tau'] = (df.Close - df.ST_MA)#/df.ST_MA
            df['LT_tau'] = (df.Close - df.LT_MA)#/df.LT_MA
            df['state'], self.state_ranges[key], trigger_state = self.get_state_representations(df.ST_tau)
            temp = df.dropna().reset_index(drop=True)
            self.train_data[key] = temp.loc[:int(0.75*len(temp))]
            self.test_data[key] = temp.loc[int(0.75*len(temp)):].reset_index(drop=True)
            self.trigger_states[key] = trigger_state
            self.transition_matrix[key] = self.compute_transition_matrix(key, lookahead=40)
            # setup a starting portfolio for the trader
            self.portfolio[key] = {'holdings': 0}
            
    def compute_EMA(self, series, num_days=50):
        """
        Computes Exponential Moving Averages of prices for every timestamp
        in the data.
        
        PARAMETERS
        ----------
        - series (pandas Series) : the (training) timeseries data
        - num_days (int, default=20) : the smoothing period
        
        RETURNS
        -------
        - A pandas series containing EMA's for every timestamp (row index).
        """
        temp = series.copy().reset_index(drop=True) # DO NOT MODIFY THE ORIGINAL DATAFRAME!
        smoothing_factor = 2 / (num_days + 1)
        EMA_prev = 0.0
        
        for idx in range(len(temp)):
            EMA_current = (temp[idx] * smoothing_factor) + EMA_prev \
                * (1 - smoothing_factor)
            # update values for next iteration
            temp[idx] = EMA_current
            EMA_prev = EMA_current 
        
        return temp
    
    def compute_momentum_signals(self, series):
        """
        Computes Moving Average Convergence Divergence and signal line
        for entries in data.
        
        PARAMETERS
        ----------
        - series (pandas Series) : the (training) timeseries data
        
        RETURNS
        -------
        - Two pandas series containing MACD and it's associated signal line 
          for every timestamp (row index).   
        """
        temp = series.copy().reset_index(drop=True) # DO NOT MODIFY THE ORIGINAL DATAFRAME!
        t1 = self.compute_EMA(temp, num_days=12)
        t2 = self.compute_EMA(temp, num_days=26)
        MACD = t1-t2
        signal_line = self.compute_EMA(MACD, num_days=9)
        return MACD, signal_line
        
    def compute_MA(self, series, long_term=True):
        """
        Computes long-term/short-term Moving Averages for the entries in 
        the data.
        
        PARAMETERS
        ----------
        - series (pandas Series) : the (training) timeseries data
        - long_term (bool, default=True) : If True, uses a 
          longer lag time (200 days)
        
        RETURNS
        -------
        - A pandas series containing the MA's for every timestamp (row index).
        """
        temp = series.copy().reset_index(drop=True) # DO NOT MODIFY THE ORIGINAL DATAFRAME!
        if long_term:
            lag = 200
        else:
            lag = 50
        assert len(temp)>lag, 'Not enough data points in this timeseries!'
        for idx in range(lag, len(temp)):
            temp[idx] = series[idx-lag:idx].mean()
        temp[:lag] = None
        return temp
    
    def map_to_states(self, series, state_ranges):
        """
        Helper function to map parameter values to states.
        
        PARAMETERS
        ----------
        - series (pandas Series) : the (training) timeseries data
        - state_ranges (dictionary) : contains the percentile ranges for every state
        
        RETURNS
        -------
        - A pandas series containing the state for every timestamp (row index).   
        """
        states = series.copy().reset_index(drop=True)
        l_st = list(state_ranges.keys())[0] # extreme left
        r_st = list(state_ranges.keys())[-1] # extreme right
        for idx, val in enumerate(series):
            # first check if value is located at the tails of the distribution
            if val<=state_ranges[l_st][0]:
                states[idx] = l_st    
            elif val>=state_ranges[r_st][0]:
                states[idx] = r_st
            else: # find the range 
                for state_idx, percentile_range in enumerate(state_ranges):
                    l, r = percentile_range[0], percentile_range[1]
                    if val>=l and val<r:
                        states[idx] = state_idx
                        break
        return states
    
    def get_state_representations(self, series, k=25):
        """
        Maps the values of the parameter of interest to states. 
        (Based on the percentiles in the distribution of the parameter.)
        
        PARAMETERS
        ----------
        - series (pandas Series) : the (training) timeseries data
        - k (int, default=True) : the number of states
        
        RETURNS
        -------
        - A pandas series containing the state for every timestamp (row index) and
          a scalar indicating the 'trigger state'.
          (point of change from 'undesirable' to 'desirable')
        """
        # first, compute the 'k' percentiles
        assert 100%k==0, 'Invalid value for the number of states. Try again with another value!'
        delta = (100/k)/100
        idxs = [delta*idx for idx in range(1,k+1)]
        percentiles = pd.Series({key+1:series.quantile(val) for \
                                 key,val in enumerate(idxs)})
        trigger_state = np.max(np.argwhere(percentiles<0))+1
        # compute the ranges for each state using the percentiles
        percentiles = pd.Series(list(zip(percentiles.shift().fillna \
                                    (series.min()), percentiles)))
        # bin values into states
        states = self.map_to_states(series, percentiles)
        return states, percentiles, trigger_state
    
    def compute_transition_matrix(self, key, lookahead=1):
        """
        Computes the state transition probabilities, between successive days.
        
        PARAMETERS
        ----------
        - key (str) : the keyword for the company's data, used in the records.
        - lookahead(int, default=1) : the number of timesteps to look ahead while 
        fetching the destination state.
        
        RETURNS
        -------
        - A numpy array of transition probabilities between states ; shape-->(k*k)
          where 'k' is the number of states.
        """
        assert key in self.train_data.keys(), \
             'Company data not found in records! Try again after storing records.'
        num_states = int(self.train_data[key].state.max()+1)
        Q = np.zeros((num_states, num_states))
        temp = self.train_data[key][:-1]
        freqs = [0.0]*num_states
        for idx in range(len(temp.index)-lookahead+1):
            current_ = self.train_data[key].iloc[idx]['state']
            next_ = self.train_data[key].iloc[idx+lookahead]['state']
            Q[int(current_), int(next_)]+=1.0
            freqs[int(current_)]+=1.0
        for idx in range(num_states):
            Q[idx,:]/= freqs[idx]
        return Q
 
    def choose_action(self, d, name):
        """
        Chooses an action for the the input observation.
        
        PARAMETERS
        ----------
        - d (pandas Series instance) : the input observation
        - name (str) : the name of the company, used while searching transition information
          
        RETURNS
        -------
        - 'buy', 'sell' or 'hold'.
        """
        # some initializations
        current_state = d.state
        caution = False
        confidence = False
        buy_rules = [0,0,0,0]
        next_vec = self.transition_matrix[name][int(current_state)]
        print(next_vec)
        num_undesirable_states = (self.trigger_states[name]+1)
        num_desirable_states = (next_vec.size-num_undesirable_states)
        if num_undesirable_states<5:
            left_basket_max = 2
        else:
            left_basket_max = num_undesirable_states//3
        if num_desirable_states<5:
            right_basket_min = next_vec.size-2
        else:
            right_basket_min = next_vec.size-num_undesirable_states//3
        # check if rules are satisfied
        # rule-1
        m1 = np.max(next_vec[:self.trigger_states[name]+1])
        m1_idx = np.argmax(next_vec[:self.trigger_states[name]+1])
        m2 = np.max(next_vec[self.trigger_states[name]+1:])
        m2_idx = np.argmax(next_vec[self.trigger_states[name]+1:])+\
                        next_vec[:self.trigger_states[name]+1].size
        if m2-m1>=0.1: # threshold
            #print('Rule #1 satisfied.')
            buy_rules[0]=1
        # rule-2
        if np.sum(next_vec[self.trigger_states[name]+1:])-\
                np.sum(next_vec[:self.trigger_states[name]+1])>=0.25: # threshold
            #print('Rule #2 satisfied.')
            buy_rules[1]=1
        # rule-3 
        if m1_idx<left_basket_max: 
            if buy_rules[0]!=1:
                caution=True
            #print('Predicted state is very undesirable.')
        # rule-3
        if m2_idx>=right_basket_min:
            if buy_rules[0]==1:
                confidence=True
            #print('Predicted state is very desirable.')
        if d.MACD>d.signal_line:
            #print('Rule #3 satisfied.')
            buy_rules[2] = True
        # sum of k most undesirable vs k most desirable
        temp_1 = np.sort(next_vec[self.trigger_states[name]+1:])
        temp_2 = np.sort(next_vec[:self.trigger_states[name]+1])
        size = 3
        if temp_1.size<size or temp_2.size<size:
            size = min(temp_1.size, temp_2.size)
        k1 = np.sum(temp_1[::-size])
        k2 = np.sum(temp_2[::-size])
        if k1-k2>0.25:
            #print('Rule #4 satisfied.')
            buy_rules[3] = True
        # finally, make a call using the rules
        if confidence or sum(buy_rules)>=3:
            return 'buy'
        elif caution or (buy_rules[0]==0 and sum(buy_rules)<=2 and m1-m2>0.05):
            return 'sell'
        else:
            return 'hold'
    
    def simulate_trader(self, override=False):
        """
        Simulates the trader's actions on the test data.
        
        PARAMETERS
        ----------
        - override (bool, default=False) : if True, previous actions will not be 
          considered while choosing current action
          
        RETURNS
        -------
        - The results of the simulation run containing the history of profits made 
          during the simulation. (dict)
        """
        results = {}
        for name in self.test_data.keys():
            profits = []
            prev_action = None
            buy_record = []
            for idx in range(len(self.test_data[name])-1):
                observation = self.test_data[name].iloc[idx]
                next_price = self.test_data[name].iloc[idx+1].Open
                action = self.choose_action(observation, name)
                if action=='buy' and self.portfolio['balance']>=(1+self.commission)*\
                    (self.transaction_volume*next_price):
                        if (prev_action=='buy' and override==True) or prev_action!='buy':
                            # buy at next day's opening price
                            self.portfolio['balance']-=(1+self.commission)*\
                                                (self.transaction_volume*next_price)
                            self.portfolio[name]['holdings']+=self.transaction_volume
                            buy_record.append(next_price)
                            prev_action = 'buy'
                elif action=='sell' and self.portfolio[name]['holdings']>=self.transaction_volume:
                        if override==True: 
                            # sell the earliest buy at next day's opening price
                            self.portfolio[name]['holdings']-=self.transaction_volume
                            profits.append(self.transaction_volume*(next_price-buy_record[0])-\
                                               (self.commission*self.transaction_volume))
                            self.portfolio['balance']+=self.transaction_volume*(next_price)-\
                                               (self.commission*self.transaction_volume)
                            buy_record.pop(0)  
                            prev_action = 'sell'
                        elif prev_action!='sell' and override==False:
                            # sell all holdings at next day's opening price
                            for b in buy_record:
                                profits.append(self.transaction_volume*(next_price-b)-\
                                               (self.commission*self.transaction_volume))
                                self.portfolio[name]['holdings']-=self.transaction_volume
                                self.portfolio['balance']+=self.transaction_volume*(next_price)-\
                                               (self.commission*self.transaction_volume)
                            assert self.portfolio[name]['holdings']==0, 'Implementation error in "sell"!'
                            buy_record = []  
                            prev_action = 'sell'
                else:# hold
                    pass
            results[name] = profits
            print(sum(profits))
        return results
                
if __name__=='__main__':        
    trader = Trader(TSLA='/Users/roshanprakash/Desktop/DeepRL_Trader/TSLA.csv')
    #print(trader.train_data['TSLA'].head())
    #print(trader.train_data['TSLA'].columns)
    print(trader.simulate_trader())
    #print(trader.transition_matrix['TSLA'])
    #print(trader.choose_action(t.train_data['TSLA'].iloc[100], 'TSLA'))
    plt.figure(figsize=(12,7))
    plt.hist(trader.train_data['TSLA'].ST_tau, bins=30, edgecolor='black', alpha=0.75)
    plt.show()