import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest, jarque_bera, chi2, norm, t
from statsmodels.tsa.stattools import adfuller
from TSA.analysis import plot_cum_per, pacf, xcorr


def whiteness_test(data, alpha=0.05, K=20, plotCumPer=True):
    """
    The function performs various whiteness tests. 
    The significance level indicates the likelihood that the signal is white but fails the test.
    The parameter K denotes the number of correlation lags used to form the Portmanteau lack-of-fit tests.
    """
    print(f"Whiteness test with {alpha*100}% significance")

    # Perform Ljung-Box-Pierce test
    S, Q, chiV = lbp_test(data, K, alpha)
    print(f"  Ljung-Box-Pierce test: {S} (white if {Q:.2f} < {chiV:.2f})")

    # Perform McLeod-Li test
    S, Q, chiV = ml_test(data, K, alpha)
    print(f"  McLeod-Li test:        {S} (white if {Q:.2f} < {chiV:.2f})")

    # Perform Monti test
    S, Q, chiV = monti_test(data, K, alpha)
    print(f"  Monti test:            {S} (white if {Q:.2f} < {chiV:.2f})")

    # Sign change test
    nRatio, confInt = count_sign_changes(data, 1-alpha)
    is_white = confInt[0] < nRatio < confInt[1]
    print(f"  Sign change test:      {is_white} (white if {nRatio:.2f} in [{confInt[0]:.2f},{confInt[1]:.2f}])")
    
    if plotCumPer: plot_cum_per(data, alpha=alpha)    


def check_if_white(data, K=20, alpha=0.05, which_test='monti', return_val=False):
    """
    The function computes the Monti test (using K terms, with default K = 20,
    and confidence alpha, with default = 0.05) to determine if the data is
    white or not, returning the decision if return_val is true.
    """

    deemed_white, Q, chiV = monti_test(data, K, alpha)

    if return_val:
        return deemed_white
    
    if deemed_white:
        print(f"The data is deemed to be WHITE according to the Monti-test (as {Q:.2f} < {chiV:.2f}).")
    else:
        print(f"The data is NOT deemed to be white according to the Monti-test (as {Q:.2f} > {chiV:.2f}).")


def monti_test(data, K=20, alpha=0.05):
    """
    The function computes the Monti statistic using K considered 
    correlations. With significance alpha, one may reject the hypothesis 
    that the residual is white if Q > chi^2_{1-alpha}(K). Thus, for a 
    95% confidence, alpha = 0.05. Unless specified, the function uses K=20 
    and alpha=0.05 as default.

    The function returns deemed_white = True if the sequence is deemed white, 
    together with the Q value and the chi2 significance level.
    """
    N = len(data)
    r = pacf(data, maxOrd=K)
    Q = N * (N + 2) * np.sum(r[1:]**2 / (N - np.arange(1, K+1)))
    chiV = chi2.ppf(1-alpha, K)
    deemed_white = Q < chiV

    return deemed_white, Q, chiV


def lbp_test(data, K=20, alpha=0.05):
    """
    Computes the modified Ljung-Box-Pierce statistic using K considered correlations.
    
    Args:
    - x: The input data.
    - K: Number of correlations to consider. Default is 20.
    - alpha: Significance level for the test. Default is 0.05.
    
    Returns:
    - deemedWhite: 1 if sequence is deemed white, 0 otherwise.
    - Q: The Q value.
    - chiV: The chi2 significance level.
    """

    x = np.atleast_2d(data)
    m, N = x.shape
    
    # Only univariate case so far
    r = xcorr(x[0], maxlag=K, biased=True)[1]
    r = r/max(r)
    r = r[K+1:2*K+1]

    Q = N * (N + 2) * np.sum(r**2 / np.arange(N-1, N-K-1, -1))
    chiV = chi2.ppf(1 - alpha, K)
    deemedWhite = Q < chiV
    return deemedWhite, Q, chiV


def ml_test(data, K=20, alpha=0.05):
    """
    Computes the McLeod-Li statistic using K considered correlations.

    Parameters:
    - x (array_like): Input sequence for which the McLeod-Li statistic is computed.
    - K (int, optional): Number of considered correlations. Default is 20.
    - alpha (float, optional): Significance level. Default is 0.05.

    Returns:
    - deemedWhite (bool): True if the sequence is deemed white.
    - Q (float): Computed Q value.
    - chiV (float): Chi2 significance level.
    """
    N = len(data)
    r = xcorr(data**2 - np.mean(data**2), maxlag=K, biased=True)[1] # Biased estimate
    r = r/max(r)
    r = r[K+1:2*K+1]

    Q = N * (N + 2) * np.sum(r**2 / np.arange(N-1, N-K-1, -1))
    chiV = chi2.ppf(1 - alpha, K)
    deemedWhite = Q < chiV
    return deemedWhite, Q, chiV


def count_sign_changes(data, confLev=0.95):
    """
    Count the number of sign changes in the data and return the ratio of sign changes.

    Parameters:
    - data: A numpy array of the data.
    - confLev: A confidence level for the whiteness test. Should be between 0 and 1. Default is 0.95.

    Returns:
    - nRatio: Ratio of sign changes.
    - confInt: Confidence interval for the resulting whiteness test (if confLev is provided).
    """
    x = np.copy(data)
    N = len(data)
    x[x > 0] = 1
    x[x <= 0] = 0
    nRatio = len(np.where(np.diff(x) != 0)[0]) / (N-1)
    
    if confLev > 1 or confLev < 0:
        raise ValueError("countSignChanges: not a legal probability.")
    
    normConfLev = norm.ppf(1 - (1 - confLev) / 2)
    confInt = ((N - 1) / 2 + normConfLev * np.sqrt((N - 1) / 4) * np.array([-1, 1])) / (N-1)
    return nRatio, confInt


def check_if_normal(data, which_test='D', alpha=0.05, return_val=False):
    """
    The function computes a normality test to determine if the data is normal
    distributed or not. If which_test is set to 'D' (default), the
    D'Agostino-Pearson's K2 test is computed, otherwise the Jarque-Bera test.
    If return_val is true, only the value is returned.
    """
    if which_test == 'D':
        _, p = normaltest(data)
        deemed_normal = p > alpha

        if return_val:
            return deemed_normal
        
        if not deemed_normal:
            print(f"The D'Agostino-Pearson K2 test indicates that the data is NOT normal distributed.")
        else:
            print(f"The D'Agostino-Pearson K2 test indicates that the data is NORMAL distributed.")

    else:
        _, p = jarque_bera(data)
        deemed_normal = p < alpha

        if return_val:
            return deemed_normal
        
        if deemed_normal:
            print(f"The Jarque-Bera test indicates that the data is NOT normal distributed.")
        else:
            print(f"The Jarque-Bera test indicates that the data is NORMAL distributed!")
    
    if return_val:
        return deemed_normal


def fisher_test(data, alpha=0.05, indF=None):
    """
    Computes the Fisher test to assess if the data has a significant periodicity.

    Parameters:
    - data (list or numpy.ndarray): Time series data for which the Fisher test is to be computed.
    - alpha (float, optional): Significance level for the test. Default is 0.05.
    - indF (int, optional): Frequency index (from 1 to floor(N/2-0.5)) for which the test is computed. 
      If not provided, the test is computed for the frequency with maximum power.

    Returns:
    - signPerFt (bool): True if the data has significant periodicity, False otherwise.
    - T (float): Test statistic representing the ratio of the squared magnitude of the FFT at a 
        frequency (either the one with maximum power or the one specified by indF) to the sum of squared magnitudes.
    - g (float): Threshold derived from the significance level alpha. Periodicity is significant if T > g.

    Raises:
    - ValueError: If provided frequency index `indF` is out of bounds.
    """

    N = len(data)
    X = np.abs(np.fft.fft(data))**2 / N

    N2 = int(np.floor(N/2 - 0.5))
    X2 = X[:N2]

    if indF is None:
        T = np.max(X2) / np.sum(X2)
    else:
        if indF < 1 or indF > N2:
            raise ValueError('fisherTest: index out of bound.')
        T = X2[indF - 1] / np.sum(X2) 

    g = 1 - (alpha / N2) ** (1 / (N2 - 1))

    signPerFt = T > g

    return signPerFt, T, g


def bolviken_test(data, Na=3, alpha=0.05):
    """
    Compute the Bolviken test to determine if data contains significant periodicities.
    
    Parameters:
    - data (numpy.ndarray): The input data.
    - Na (int): A parameter for the test, sensitivity (?)
    - alpha (float, optional): Significance level. Default is 0.05.
    
    Returns:
    - signPerInd (numpy.ndarray): Indices of the frequency found to be significant.
    - signPerVal (numpy.ndarray): Decision values for the significant indices.
    """
    
    N = len(data)
    N2 = int(np.floor(N/2 - 0.5))
    
    rho2 = np.abs(np.fft.fft(data))**2 / N
    rho2 = rho2[:N2]
    ordR = np.sort(rho2)
    sigmaEst = np.sum(ordR[:N2-Na])
    
    signPerValVec = np.zeros(N2)
    for k in range(len(rho2)):
        ratioT = rho2[k] / sigmaEst
        prodQ = 1
        for k2 in range(1, N2-Na+1):
            prodQ = prodQ / (1 + k2 * ratioT / (Na + k2 - 1))
        signPerValVec[k] = prodQ * N2
        
    signPerInd = np.where(signPerValVec < alpha)[0]
    signPerVal = signPerValVec[signPerInd]
    
    return signPerInd, signPerVal


def test_mean(data, mean0=0, signLvl=0.05, return_val=False):
    """"
    The function tests if the mean of the data can be deemed to be that of a 
    process with mean-value mean0 (default zero), with significance signLvl 
    (default 0.05), returning the decision, as well as the test ratio and the  
    decision limit.
    """
    x = np.copy(data)
    N = len(x)

    tLimit = t.ppf(1 - signLvl / 2, N - 1) ** 2
    tRatio = N * (np.mean(x) - mean0) ** 2 / np.var(x, ddof=1)

    rejectMean = tRatio > tLimit
    if not rejectMean:
        print(f'Mean of data IS deemed to be {mean0}!')
    else:
        print(f'Mean of data is NOT deemed to be {mean0}.')

    if return_val:
        return rejectMean, tRatio, tLimit


def check_stationarity(data, maxLag=None, signLvl=0.05, return_val=False):
    """
    Check the stationarity of a time series data using the Augmented Dickey-Fuller test.
    
    Parameters:
    - data (array-like): The time series data to be tested.
    - maxLag (int, optional): The maximum number of lags to be considered for the test. Default is None.
    - signLvl (float, optional): The significance level for the test. Default is 0.05.
    - return_val (bool, optional): If True, the function returns the stationarity status, test statistic, and p-value. Otherwise, it prints the results. Default is False.
    
    Returns:
    - tuple: A tuple containing the stationarity status (True if stationary, False otherwise), test statistic, and p-value if return_val is set to True.
    """
    T, p = adfuller(data, maxlag=maxLag)[:2]

    deemed_stationary = p < signLvl

    if return_val:
        return deemed_stationary, T, p
    
    if deemed_stationary:
        print(f'Data is deemed STATIONARY according to the Augmented Dickey-Fuller test as p={round(p,3)} < {signLvl}.')
    else:
        print(f'Data is deemed NOT stationary according to the Augmented Dickey-Fuller test as p={round(p,3)} > {signLvl}.')
