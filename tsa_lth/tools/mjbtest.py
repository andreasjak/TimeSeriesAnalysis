# Author: Torbjörn Onshage
import numpy as np
import scipy as sp

def mjbtest( data, signLvl=0.05 ):
    """
    Computes of the multivaraite version of the Jarque-Bera test of 
    normality with significance signLvl (default 0.05).  

    This is a Python implementation of the provided MatLab function 
    with the same name.

    Reference: 
        "An Introduction to Time Series Modeling" by 
        Andreas Jakobsson Studentlitteratur, 2013
    
    Parameters:
    - data: Sample Matrix of dimension (nbr_features, nbr_samples), 
            where each column contains a multivariate sample.

    Returns: 
    - isNormal (bool): Boolean result of the normality test.
    - p_val (float): Resulting P-value of the test.
    """
    # Assert valid argument: data
    if (len(data.shape) == 1): 
        nbr_features = 1
        nbr_samples = data.shape[0]
    elif (len(data.shape) == 2):
        nbr_features = data.shape[0]
        nbr_samples = data.shape[1]
        if (nbr_features > nbr_samples): 
            raise ValueError('Warning: mjbtest: nbr_features (data.shape[0]) is larger than ' +
                             'nbr_samples (data.shape[1])')
    else:
        raise ValueError('mjbtest: argument data must be 2-dimensional with each column ' +
                         'containing a (multivariate) sample, i.e. data.shape = (nbr_features, nbr_samples)')

    # Assert valid argument: signLvl
    if (signLvl <= 0 or signLvl >= 1): 
        raise ValueError('mjbtest: significance level must be between 0 and 1')

    assert len(data.shape) == 2

    # Form mx (mean of data) and Sx (Covariance Matrix of data)
    mx = np.mean( data, axis=1 )
    Sx = np.cov( data, ddof=1 ) # Normalised by (N-1)   

    Ps = np.linalg.inv( np.linalg.cholesky( Sx ) )
    
    v_mat = Ps @ ( data - mx.reshape(-1, 1) )

    b1 = np.sum( np.power(v_mat, 3), axis=1 ) / nbr_samples
    b2 = np.sum( np.power(v_mat, 4), axis=1 ) / nbr_samples

    # Form test statistics, lambda_s and lambda_k, and form the test.
    lambda_s = nbr_samples * np.dot(b1,b1) / 6
    lambda_k = nbr_samples * np.dot(b2-3,b2-3) / 24

    # Joint test statistics.
    p_val = 1 - sp.stats.chi2.cdf( lambda_s+lambda_k, 2*nbr_features ) 

    isNormal = p_val > signLvl

    return isNormal, p_val