# coding: utf-8
import numpy as np
from pandas import *
import ystockquote
import datetime
import math as m
import numpy as np
from memoize import Memoizer

class ep_risk:
    #example folio_dict: {’JNJ’:10**6, ’AMGN’:10**6, ’UNH’:10**6, ’MYL’:10**6, ’A’: 10**6, ’XLV’: -(5*10**6)}
    #hold_per is equivalent to 
    def __init__(self, folio_dict):
        self.port= folio_dict

    memo = Memoizer({})
    #caches the last data pull
    @memo(max_age=60)
    def get_data(self, end_date = "01/15/15", window_length = 700):
        last_date = datetime.datetime.strptime(end_date, "%x")
        first_date = last_date-datetime.timedelta(days=window_length)
        s= first_date.__str__()
        l= last_date.__str__()

        #Create List of Dataframes for all stocks
        df_s = [self.prices(tick,s,l) for tick in self.port.keys()]
        #Merge that List of Dataframes
        df_final = df_s[0]
        for i in range(1,len(df_s)):
            df_final = DataFrame.join(df_final,df_s[i])

        return df_final

    def prices(self,ticker,start_string,end_string):
        #holds key:Date Val:Adj Close
        data_hash = dict()
        #Create dictionary from yahoo data
        tick_test = ystockquote.get_historical_prices(ticker,  start_string, end_string)
        #fill data_hash
        for date in tick_test.keys():
            data_hash[datetime.date(int(date[:4]), int(date[5:7]), int(date[8:]))] = float(tick_test[date]['Adj Close'])
        #make datahash into dataframe called df
        df = DataFrame.from_dict(data_hash, orient='index', dtype=float)
        #rename first column
        df.columns = [ticker]
        #sort in date order
        df = df.sort(ascending=True, inplace=False)
        return df

    def data_stats(self, n_days=2, end_date = "01/15/15", window_length = 700):
        price_data = self.get_data(end_date = end_date, window_length = window_length)
        #calculate n-day returns
        returns = price_data.pct_change(periods=n_days, fill_method='pad', limit=None, freq=None)
        #apply log transformatinon
        returns = 1+returns
        returns = returns.applymap(m.log)

        #calculate means and volatilities
        means = returns.mean(axis=0)
        vols = returns.std(axis=0)

        #calc correlations
        correlation_matrix = returns.corr()

        #returns lambda_plus for PCA approaches
        lambda_plus = (1 + m.sqrt(float(len(self.port.keys()))/max([returns.ix[:,i].count() for i in range(0,len(self.port.keys()))])))**2

        return correlation_matrix, means, vols, lambda_plus

    def pca_risk(self, alpha, n_days=2, end_date = "01/15/15", window_length = 700):
        correlation_matrix, means, vols, lambda_plus = self.data_stats(n_days=n_days, end_date = end_date, window_length = window_length)
        #return eigenvectors and values (eigenvectors are columns)
        e_vals, e_vecs = np.linalg.eigh(correlation_matrix)
        #sort (largest to smallest)
        indices = e_vals.argsort()[::-1]
        sorted_evals = [e_vals[i] for i in indices]
        sorted_evecs = e_vecs[:, indices]

        significant_num = len([value for value in sorted_evals if value > lambda_plus])
        cleaned = self.clean_correlation_matrix(significant_num, sorted_evals, sorted_evecs)
        cleaned_df = DataFrame(data=cleaned, index=correlation_matrix.columns.tolist(), columns=correlation_matrix.columns.tolist(), dtype=float, copy=True)

        var,es = self.mc_risk(alpha,cleaned_df,means,vols)
        return var,es
    
    def unclean_risk(self, alpha, n_days=2, end_date = "01/15/15", window_length = 700):
        correlation_matrix, means, vols, lambda_plus = self.data_stats(n_days=n_days, end_date = end_date, window_length = window_length)
        var,es = self.mc_risk(alpha,correlation_matrix,means,vols)
        return var,es

    
    def historical_risk(self, alpha,n_days=2, end_date = "01/15/15", window_length = 700):
        price_data = self.get_data(end_date = end_date, window_length = window_length)
        returns = price_data.pct_change(periods=n_days, fill_method='pad', limit=None, freq=None)
        return self.port_rets(returns, alpha)
        
    def mc_risk(self, alpha, corr_df, means, vols):
        #generate 50,000 samples from T dist
        joint_scenarios = self.multivariatet_sampler(np.zeros(len(self.port.keys())),corr_df,4,50000)
        #scale these scenarios by multiplying by scaling vol then shifting by mean
        joint_scenarios = m.sqrt(1/2.)*np.dot(joint_scenarios, np.diag(vols)) + np.tile(means,(50000,1))
        #calculate portfolio returns
        var_alpha, es_alpha = self.port_rets(joint_scenarios,alpha)
        return var_alpha, es_alpha

    def port_rets(self, scenarios,alpha):
        port_vec = np.array(self.port.values())
        sim_returns = np.dot(scenarios, port_vec)
        var_alpha = np.percentile(sim_returns, alpha)
        es_alpha = np.mean([v for v in sim_returns.tolist() if v < var_alpha])
        return var_alpha, es_alpha

    def clean_correlation_matrix(self, significant_num, lambdas, vecs):
        dimension = len(lambdas)
        matrix = np.zeros((dimension,dimension))
        for i in range(0,dimension):
            for j in range(0,dimension):
                ro = 0
                for k in range(0,significant_num):
                    ro += lambdas[k] * vecs[:,k][i] * vecs[:,k][j]
                if (i ==j):
                    eii_sq = 1
                    for k in range(0,significant_num):
                        eii_sq = eii_sq - lambdas[k]*vecs[:,k][i]*vecs[:,k][i]
                    ro = ro + eii_sq
                matrix[i,j] = ro
        return matrix


    #Known Function to generate samples from multivariate t in python (Kenny Chowdhary - PhD)
    def multivariatet_sampler(self,mu,Sigma,N,M):
        '''
        Output:
        Produce M samples of d-dimensional multivariate t distribution
        Input:
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        N = degrees of freedom
        M = # of samples to produce
        '''
        d = len(Sigma)
        g = np.tile(np.random.gamma(N/2.,2./N,M),(d,1)).T
        Z = np.random.multivariate_normal(np.zeros(d),Sigma,M)
        return mu + Z/np.sqrt(g)


