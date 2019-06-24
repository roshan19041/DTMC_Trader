# DTMC_Trader

A rule-based trading algorithm based on Discrete-Time-Markov-Chains. 

For any given day, the value of a stock's (Daily Closing Price - Short term Moving Average) is indicative of how well the stock has performed on that day. Higher values would essentially imply higher buy rating for the stock. This parameter's value was computed for every day in the historical time-series data for every stock in the portfolio and then the values were mapped to some 'k' number of states based on percentiles. So, the states which corresponded to higher values were certainly more desirable to buy in than those that corresponded to lower values. A transition probability matrix for state transitions was computed for each stock in the portfolio.

The trading algorithm is constrained by a set of buy/hold/sell rules built on top of the transition probability matrices and other technical indicators. For any day, the trader looks at some of the most desirable and least desirable states that the environment is going to transition to and then chooses a rule-driven/heuristics-based action.
