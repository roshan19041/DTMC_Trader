# DTMC_Trader

A rule-based trading algorithm based on Discrete-Time-Markov-Chains. 

The algorithm is constrained by a set of buy/hold/sell rules built on top of the transition probability matrix and other technical indicators.

The higher the values of the parameter of interest (Daily Closing Price-MA) that fall into a state, the greater the desirability of being in that state. The trader looks at some of the most desirable and least desirable states that the environment is going to transition to and then makes a transaction.

