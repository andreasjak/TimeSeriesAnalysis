# Author: Filipp Lernbo
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest, jarque_bera, chi2, norm, t
from statsmodels.tsa.stattools import adfuller
from tsa_lth.analysis import plot_cum_per, pacf, xcorr


def whiteness_test(data, alpha=0.05, K=25, plotCumPer=False):
    """
    Conducts whiteness tests on time series data to check for randomness.

    This function applies Ljung-Box-Pierce, McLeod-Li, Monti, and sign change tests on 'data'.
    The tests are based on autocorrelations up to lag 'K' and a significance level of 'alpha'.
    If 'plotCumPer' is True, a cumulative periodogram is also displayed.

    Parameters:
    - data (array-like): Time series data to be tested.
    - alpha (float, optional): Significance level for the tests, default is 0.05.
    - K (int, optional): Number of lags in autocorrelation for the tests, default is 20.
    - plotCumPer (bool, optional): Whether to plot the cumulative periodogram, default is True.

    Returns:
    None: Results are printed to the console; see individual test functions for potential return values.
    """
    print(f"Whiteness test with {alpha*100}% significance")

    # Perform Ljung-Box-Pierce test
    S, Q, chiV = lbp_test(data, K, alpha)
    print(f"  Ljung-Box-Pierce test: {S} (white if {Q:.2f} < {chiV:.2f})")

    # Perform McLeod-Li test
    S, Q, chiV = ml_test(data, K, alpha)
    print(f"  McLeod-Li test:        {S} (white if {Q:.2f} < {chiV:.2f})")

    # Perform Monti test
    S, Q, chiV = monti_test(data, K, alpha, return_val=True)
    print(f"  Monti test:            {S} (white if {Q:.2f} < {chiV:.2f})")

    # Sign change test
    nRatio, confInt = count_sign_changes(data, 1-alpha)
    is_white = confInt[0] < nRatio < confInt[1]
    print(f"  Sign change test:      {is_white} (white if {nRatio:.2f} in [{confInt[0]:.2f},{confInt[1]:.2f}])")
    
    if plotCumPer: plot_cum_per(data, alpha=alpha)


def monti_test(data, K=20, alpha=0.05, return_val=True):
    """
    Conducts the Monti test to assess if a data sequence is white noise.

    Args:
    data (array-like): Input sequence of time series data.
    K (int, optional): Number of autocorrelations used in the test. Defaults to 20.
    alpha (float, optional): Significance level for the test. Defaults to 0.05.
    return_val (bool, optional): If True, returns a tuple instead of printing the result. Defaults to True.

    Returns:
    tuple (bool, float, float), optional: A tuple (deemed_white, Q, chiV) indicating the test result (if return_val is True).
                                           - deemed_white: True if data is considered white noise.
                                           - Q: Monti statistic.
                                           - chiV: Chi-squared distribution critical value.

    Note: If return_val is False, the test result is printed instead.
    """
    N = len(data)
    r = pacf(data, maxOrd=K)
    Q = N * (N + 2) * np.sum(r[1:]**2 / (N - np.arange(1, K+1)))
    chiV = chi2.ppf(1-alpha, K)
    deemed_white = Q < chiV

    if return_val:
        return deemed_white, Q, chiV

    if deemed_white:
        print(f"The data is deemed to be WHITE according to the Monti-test (as {Q:.2f} < {chiV:.2f}).")
    else:
        print(f"The data is NOT deemed to be white according to the Monti-test (as {Q:.2f} > {chiV:.2f}).")


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
    Performs a normality test on data, using either D'Agostino-Pearson's K2 or Jarque-Bera test.

    Args:
    data (array-like): Dataset for normality testing.
    which_test (str, optional): Specifies the test to use ('D' for D'Agostino-Pearson, any other value for Jarque-Bera). Defaults to 'D'.
    alpha (float, optional): Significance level for the normality test. Defaults to 0.05.
    return_val (bool, optional): If True, only the test decision is returned. Otherwise, the decision is printed.

    Returns:
    bool: If return_val is True, returns whether the data is deemed normal. Otherwise, prints the test outcome.

    Note: The function compares p-value with alpha to determine the normality of the data and informs the user accordingly.
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
    """
    Tests if the sample mean significantly differs from mean0.

    Args:
    data (array-like): Sample data points.
    mean0 (float, optional): Hypothesized mean value. Defaults to 0.
    signLvl (float, optional): Significance level for the test. Defaults to 0.05.
    return_val (bool, optional): If True, returns a tuple of (decision, test ratio, decision limit).

    Returns:
    bool: If return_val is False, returns whether the null hypothesis is rejected.
    tuple: If return_val is True, returns (rejectMean, tRatio, tLimit).

    Note: Prints the decision outcome.
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
