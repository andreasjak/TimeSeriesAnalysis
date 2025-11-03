#Marcus Lindell & Casper Schwerin
import numpy as np
from scipy.stats import t
from scipy.stats import f

def testMean(data, mean0=None, signLvl=0.05):
    '''
    The function tests if the mean of the data can be deemed to be that of a 
    process with mean-value mean0 (default zero), with significance signLvl 
    (default 0.05), returning the decision, as well as the test ratio and the  
    decision limit (for univariate data this is t-distributed, for multi-variate, 
    it is Hotelling-T2 distributed).

    The function assumes that the data is  Gaussian distributed, and works 
    for both univariate and multivariate data sets, with the latter assumed 
    to consist of N row vectors, formed as an N x m matrix. 


    Reference: 
    "An Introduction to Time Series Modeling" by Andreas Jakobsson
    Studentlitteratur, 2013
    '''
    N,m = data.shape

    if (N==1) and (m>1):     # Univariate data given as row vector.
        N = m
        m = 1
    
    if mean0 is None:    
        mean0 = np.zeros((1, m))


    if m == 1:                # Form squared t-ratio and the rejection limit.
        tLimit = np.power(t.ppf(1-signLvl/2, df=N-1), 2)
        tRatio = N*(np.mean(data) - mean0) ** 2 / np.var(data)
    
    else:                     # Form Hotelling T2 and the rejection limit.
        R0 = _covMvect( data, mean0 )
        R1 = _covMvect( data, np.mean(data))
        tLimit = (N-1)*m/(N-m) * f.ppf( 1-signLvl, m, N-m)
        tRatio = (N-1)*np.divide(np.linalg.det( R0 ), np.linalg.det( R1 )) - (N-1)
 
    rejectMean = tRatio > tLimit # if true --> suggests there is a trend

    return (rejectMean, tRatio, tLimit)

def _covMvect(data, meanD=None, fb=False):
    '''
    The function computes the unbiased covariance matrix estimate of the data, 
    which is assumed to consist of N row vectors, formed as an N x m matrix. 
    The data is assumed to have a 1 x m mean vector meanD; if not given, it is 
    estimated. If fb is set, forward-backward averaging is used.


    Reference: 
    "An Introduction to Time Series Modeling" by Andreas Jakobsson
    Studentlitteratur, 2013
    '''

    N,m = data.shape

    if meanD is None:
        meanD = np.mean(data) # If not specified, estimate the mean.

    else:
        n1, m1 = meanD.shape # Check dimensionality
        if(n1 == m) and (m1 == 1): # If flipped, transpose to row vector.
            meanD = meanD.T
            m1 = n1
            n1 = 1

        assert m1 == m

    data = np.subtract(data, np.ones(N,1)*meanD) # Compensate for mean value
    R = np.divide(np.multiply(data.T, data), N-1)

    if fb:
        H = np.fliplr(np.identity(m))
        R = (R + np.multiply(np.multiply(H, R.T), H)) / 2

    return (R, meanD)