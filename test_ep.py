# coding: utf-8
import random
from ep_sim import *

#make test portfolio
some_tickers = ['JNJ', 'AAPL','MRK', 'GILD', 'AMGN', 'BMY', 'BIIB', 'ABBV', 'UNH', 'CELG', 'ESRX', 'LLY', 'MDT', 'ABT', 'TMO', 'MCK', 'ACT', 'BAX', 'AGN', 'ALXN', 'COV', 'WLP', 'AET']
some_amts = [random.randrange(0, 10**6) for t in some_tickers]

port_d = dict(zip(some_tickers, some_amts))
portfolio = ep_risk(port_d)

#example calls
print "(Var,ES) under alpha=1"
print "1. PCA Approach: ", portfolio.pca_risk(1, n_days=2, end_date = "01/15/15", window_length = 300)
print "2. Uncleaned Corr: ",portfolio.unclean_risk(1, n_days=2, end_date = "01/15/15", window_length = 300)
print "3. Historical Risk", portfolio.historical_risk(1, n_days=2, end_date = "01/15/15", window_length = 300)