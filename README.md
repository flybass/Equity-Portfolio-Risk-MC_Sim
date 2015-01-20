# Equity-Portfolio-Risk-MC_Sim

Takes a basket of equities with tickers, $ amount positions, data length (size of historical window), and days for returns (ie 2-day returns).  Downloads historical price data from yahoo finance.  This module is based from class notes and work from Professor Marco Avellaneda (http://math.nyu.edu/faculty/avellane/).  Specifically, one can reference this paper http://www.finance-concepts.com/press/JeongMarco0630-1.pdf.


Yields 3 types of VaR and ES estimates.  

1.  PCA to Extract Significant Eigenvalues (returns T-distributed with 4 dof)
2.  Uncleaned Correlation Matrix (returns T-distributed with 4 dof)
3.  Historical - calculated from historical returns on given portfolio

