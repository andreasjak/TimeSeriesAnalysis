import numpy as np
import pandas as pd
import re
import scipy
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.signal import correlate, correlation_lags
import statsmodels.tsa.stattools as stats

def autolags(y):
    """
    Determines a reasonable amount of lags to use for acf and pacf plots based on the amount of data.
    
    Parameters:
    - y: Data to calculate amount of lags from.
    """
    return int(round( 15*(len(y)/100)**0.25 ))

def acf(y, maxOrd='auto', signLvl=0.05, plotIt=False, maOrder=0, includeZeroLag=True):
    """
    Computes the auto-correlation function (ACF) of a given time series.

    Parameters:
    - y (array-like): Time series data.
    - maxOrd (int or str): Maximum order for the ACF. Can be 'auto', 'full', or an integer.
    - signLvl (float): Significance level for confidence intervals, default is 0.05.
    - plotIt (bool): If True, plots the ACF. Default is False.
    - maOrder (int): Order of the moving average process, used for confidence bands.
    - includeZeroLag (bool): If True, includes the zero lag in the ACF calculation. Default is True.

    Returns:
    - rho (array): ACF values up to the specified maxOrd.
    """
    if maxOrd == -1 or maxOrd == 'full': # whole dataset
        maxOrd = len(y)-1
    elif maxOrd == 'auto': # automatic trim
        maxOrd = autolags(y)

    if maOrder > maxOrd:
        raise ValueError('ACF: call with maOrder larger than maxOrd.')
    if not 0 <= signLvl <= 1:
        raise ValueError('ACF: not a valid level of significance.')
    
    signScale = scipy.stats.norm.ppf(1 - signLvl / 2, 0, 1)
    rho, _ = stats.acf(y, nlags=maxOrd, alpha=signLvl, fft=False)

    if includeZeroLag:
        rangeLags = np.arange(0, maxOrd + 1)
        maxRange = 1.1
        startLag = 0
    else:
        rangeLags = np.arange(1, maxOrd + 1)
        rho = rho[1:]
        maxRange = max(np.abs(rho)) * 1.2
        startLag = 1

    if plotIt:
        plt.stem(rangeLags, rho)
        plt.xlabel('Lag')
        plt.ylabel('Amplitude')
        plt.title('ACF')
        condInt = signScale * np.ones(len(rangeLags)) / np.sqrt(len(y))
        condInt = condInt * np.sqrt(1 + 2 * np.sum(rho[:maOrder]**2))
        plt.plot(rangeLags, condInt, 'r--')
        plt.plot(rangeLags, -condInt, 'r--')
        plt.axis([startLag, len(rangeLags) - 1, -maxRange, maxRange])
        plt.grid()

    return rho

def pacf(y, maxOrd='auto', signLvl=0.05, plotIt=False, includeZeroLag=True):
    """
    Computes the partial auto-correlation function (PACF) of a given time series.

    Parameters:
    - y (array-like): Time series data.
    - maxOrd (int or str): Maximum order for the PACF. Can be 'auto', 'full', or an integer.
    - signLvl (float): Significance level for confidence intervals, default is 0.05.
    - plotIt (bool): If True, plots the PACF. Default is False.
    - includeZeroLag (bool): If True, includes the zero lag in the PACF calculation. Default is True.

    Returns:
    - phi (array): PACF values up to the specified maxOrd.
    """
    if maxOrd == -1 or maxOrd == 'full': # Whole dataset
        maxOrd = len(y)-1
    elif maxOrd == 'auto': # Automatic trim
        maxOrd = autolags(y)

    if not 0 <= signLvl <= 1:
        raise ValueError('PACF: not a valid level of significance.')
    
    signScale = scipy.stats.norm.ppf(1 - signLvl / 2, 0, 1)
    y = y - np.mean(y)
    phi = stats.pacf_yw(y, nlags=maxOrd, method='mle')
    phi[1:] = -phi[1:]  # Negate all but the first value to match MATLAB's output

    if includeZeroLag:
        rangeLags = np.arange(0, maxOrd + 1)
        maxRange = 1.1
        startLag = 0
    else:
        rangeLags = np.arange(1, maxOrd + 1)
        phi = phi[1:]
        maxRange = max(np.abs(phi)) * 1.2
        startLag = 1

    if plotIt:
        plt.stem(rangeLags, phi)
        plt.xlabel('Lag')
        plt.ylabel('Amplitude')
        condInt = signScale * np.ones(len(rangeLags)) / np.sqrt(len(y))
        plt.plot(rangeLags, condInt, 'r--')
        plt.plot(rangeLags, -condInt, 'r--')
        plt.axis([startLag, len(rangeLags) - 1, -maxRange, maxRange])
        plt.grid()
        # plt.show()

    return phi

def tacf(y, maxOrd='auto', alpha=0.02, signLvl=0.05, plotIt=False, includeZeroLag=True, titleStr=None):
    """
    Estimate the alpha-trimmed ACF of y for lags up to maxOrd.

    Parameters:
    - y: Data series to calculate TACF on.
    - maxOrd: Maximum lag order to calculate the ACF. Can be 'full' for whole dataset or 'auto' to determine automatically.
    - alpha: Trim level, set to 1-2% for general time series, 3-5% for medium contaminated series, and 6-10% for heavily contaminated series. Default is 2%.
    - signLvl: Significance level for the Gaussian confidence interval, default is 0.05.
    - plotIt: If True, the ACF will be plotted.
    - includeZeroLag: If True (default), the tacf is from lag 0 to maxOrd, otherwise from lag 1.
    - titleStr: Additional string to be included in the plot title.

    Returns:
    - rho: Estimated TACF values.
    """
    y = y.copy()
    if maxOrd == -1 or maxOrd == 'full': # Whole dataset
        maxOrd = len(y)
    elif maxOrd == 'auto': # Automatic trim
        maxOrd = autolags(y)

    signScale = scipy.stats.norm.ppf(1 - signLvl / 2, 0, 1)
    N = len(y)
    ys = np.sort(y)

    # Form trimmed data
    g = N * alpha
    indY = y < ys[int(np.floor(g))]
    y[indY] = 0
    indY = y > ys[int(np.floor(N - g))]
    y[indY] = 0

    # Find kept indices
    La = (y > ys[int(np.floor(g))]) & (y < ys[int(np.floor(N - g))])
    Lg = np.zeros(N)
    Lg[La] = 1

    # Form trimmed mean estimate
    my = np.sum(y) / np.sum(Lg)

    # Estimate TACF
    rho = xcorr(Lg * (y - my), maxlag=maxOrd)[1]
    g2 = xcorr(Lg, maxlag=maxOrd)[1]
    rho = rho[maxOrd:] / g2[maxOrd:]
    rho = rho / rho[0]

    if includeZeroLag:
        rangeLags = np.arange(0, maxOrd + 1)
        maxRange = 1.1
        startLag = 0
        g2 = g2[maxOrd:]
    else:
        rangeLags = np.arange(1, maxOrd + 1)
        rho = rho[1:]
        g2 = g2[maxOrd + 1:]
        maxRange = np.max(np.abs(rho)) * 1.2
        startLag = 1

    # Display results
    if plotIt:
        plt.stem(rangeLags, rho)
        plt.xlabel('Lag')
        plt.ylabel('Amplitude')
        condInt = signScale * np.ones(len(rangeLags)) / np.sqrt(g2)
        plt.plot(rangeLags, condInt, 'r--')
        plt.plot(rangeLags, -condInt, 'r--')
        plt.axis([startLag, len(rangeLags) - 1, -maxRange, maxRange])
        plt.title(f'TACF ({titleStr})') if titleStr else plt.title('TACF')
        plt.grid()
        plt.show()

    return rho

def plotACFnPACF(y, noLags='auto', titleStr=None, signLvl=0.05, realis=False, includeZeroLag=True, return_val=False):
    """
    Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of the given data.

    Parameters:
    - y (list/array-like): The dataset for which ACF and PACF needs to be plotted.
    - noLags (int, optional): Number of lags to be considered. Defaults to 'auto'.
    - titleStr (str, optional): A string to be appended in the title of plots. Defaults to an empty string.
    - signLvl (float, optional): Significance level for the ACF and PACF plots. Defaults to 0.05.
    - realis (bool, optional): If set to True, it will plot the realisation of the data. Defaults to False.
    - includeZeroLag (bool): If True, includes the zero lag in the ACF and PACF plots. Default is True.
    - return_val (bool, optional): If set to True, returns computed ACF and PACF values. Defaults to False.

    Returns:
    - tuple: If return_val is True, returns a tuple of the estimated ACF and PACF values. Otherwise, nothing is returned.
    """
    if realis:
        plt.figure(figsize=(6, 9))
        pl = 3
        i = 1
        plt.subplot(pl, 1, i)
        plt.plot(y)
        plt.title(f'Realisation ({titleStr})') if titleStr else plt.title('Realisation')
        plt.grid()

    else:
        plt.figure(figsize=(6, 6))
        pl = 2
        i = 0

    plt.subplot(pl, 1, 1+i)
    acfEst = acf(y, noLags, signLvl, plotIt=True, includeZeroLag=includeZeroLag)
    plt.title(f'ACF ({titleStr})') if titleStr else plt.title('ACF')
    
    plt.subplot(pl, 1, 2+i) 
    pacfEst = pacf(y, noLags, signLvl, plotIt=True, includeZeroLag=includeZeroLag)
    plt.title(f'PACF ({titleStr})') if titleStr else plt.title('PACF')
    
    plt.tight_layout()
    plt.show()
    
    if return_val:
        return acfEst, pacfEst

    pass

def plot_cum_per(x, alpha=0.05, plotIt=True, titleStr=None, return_val=False):
    """
    Computes and optionally plots the cumulative periodogram of the given signal.

    Parameters:
    - x (numpy.array): Input time series or signal.
    - alpha (float, optional): Significance level for the confidence interval. Defaults to 0.05. 
                               Supported values: 0.01, 0.05, 0.10, 0.25.
    - plotIt (bool, optional): Flag to decide whether to plot the result. Defaults to False.
    - titleStr (str, optional): A string to be appended in the title of plots. Defaults to an empty string.
    - return_val (bool, optional): If set to True, returns computed periodogram values. Defaults to False.

    Returns:
    - tuple: Cumulative periodogram (C), lower (x1) and upper (x2) confidence intervals, and confidence interval multiplier (Ka). 
             Ka is 0 if `alpha` is unsupported.
    """
    
    N = len(x)
    X = np.abs(np.fft.fft(x))**2
    S = np.sum(X)/2

    n = N//2
    C = np.zeros(n)
    for k in range(n):
        C[k] = np.sum(X[:k+1]) / S

    if alpha == 0.01:
        Ka = 1.63
    elif alpha == 0.05:
        Ka = 1.36
    elif alpha == 0.10:
        Ka = 1.22
    elif alpha == 0.25:
        Ka = 1.02
    else:
        Ka = 0
        C = np.array([-1])

    x1 = (np.arange(n)/n) - Ka / np.sqrt((n-1)//2)
    x2 = (np.arange(n)/n) + Ka / np.sqrt((n-1)//2)

    if plotIt and not np.any(C == -1):
        ff = 0.5*np.arange(n)/n
        plt.plot(ff, C)
        plt.plot(ff, x1, 'r--')
        plt.plot(ff, x2, 'r--')
        plt.axis([0, 0.5, 0, 1])
        plt.xlabel('Frequency')
        plt.ylabel('C(w)')
        plt.grid()
        plt.title(f'Cumulative Periodogram ({titleStr})') if titleStr else plt.title('Cumulative Periodogram')
        plt.show()

    if return_val:    
        return C, x1, x2, Ka

def xcorr(x,y=None, maxlag=None, biased=False):
    """
    Compute the cross-correlation between two signals.

    Parameters:
    - x (array-like): 1st signal.
    - y (array-like, optional): 2nd signal. If None, cross-correlation of x with itself is computed.
    - maxlag (int, optional): Maximum lag to consider. If specified, result is centered around zero lag.
    - biased (bool): If True, normalize the result by the length of x. Default is False.
    - plotIt (bool): Currently unused.

    Returns:
    - lags (array-like): Lags of correlation.
    - corr (array-like): Cross-correlation coefficients.
    """
    
    if y is None: y=x

    corr = correlate(x, y)
    lags = correlation_lags(len(x), len(y))

    if maxlag:
        mid = len(corr) // 2
        corr = corr[mid-maxlag:mid+maxlag+1]
        lags = lags[mid-maxlag:mid+maxlag+1]

    if biased:
        corr = corr/len(x)

    return lags, corr

def ccf(y1, y2, numLags=None, titleStr=None , N=None, plotIt=True):
    """
    Compute the normalized cross-correlation between two signals and optionally visualize the result.

    Parameters:
    - y1, y2 (array-like): Input signals whose cross-correlation is to be calculated. Both arrays should be real sequences.
    - numLags (int, optional): Number of lags to compute for the cross-correlation function. Defaults to the minimum length of y1, y2 minus one.
    - titleStr (str, optional): Title for the plot when 'plotIt' is True. If None and 'plotIt' is True, a default title is used.
    - N (int, optional): Used for confidence interval calculation in the plot. If None, defaults to the maximum length of y1, y2.
    - plotIt (bool): If True, a plot of the cross-correlation function is displayed. Default is True.

    Returns:
    - xcf (array-like): Normalized cross-correlation function values (only if 'plotIt' is False). Ranges from -numLags to numLags.
    """
    y1 = y1.copy()
    y2 = y2.copy()
    
    L1, L2 = len(y1), len(y2)
    if numLags is None:
        numLags = min(L1,L2)-1
    numLags = min(numLags, min(L1,L2)-1)
    if N is None:
        N = max(L1,L2)

    # Subtract the mean from both sequences
    y1 = y1 - np.mean(y1)
    y2 = y2 - np.mean(y2)
    
    # Equalize the lengths of the sequences
    if L1 > L2:
        y2 = np.pad(y2, (0, L1-L2), 'constant')
    elif L1 < L2:
        y1 = np.pad(y1, (0, L2-L1), 'constant')
    
    # Compute the FFT of both sequences
    nFFT = 2**(np.ceil(np.log2(max([L1, L2]))).astype(int) + 1)
    F_y1 = np.fft.fft(y1, nFFT)
    F_y2 = np.fft.fft(y2, nFFT)
    
    # Compute the autocorrelation of each sequence
    ACF1 = np.fft.ifft(F_y1 * np.conj(F_y1)).real
    ACF2 = np.fft.ifft(F_y2 * np.conj(F_y2)).real
    
    # Compute the cross-correlation using the FFT
    xcf = np.fft.ifft(F_y1 * np.conj(F_y2)).real
    xcf = np.hstack([xcf[-(numLags+1):], xcf[:numLags+1]])  # taking values from the end and start of the array
    
    # Normalize the cross-correlation
    xcf = xcf / (np.sqrt(ACF1[0]) * np.sqrt(ACF2[0]))
    xcf = xcf[-1:0:-1]

    if plotIt:
        rangeLags = np.arange(-numLags, numLags+1, 1)
        _,_,baseline = plt.stem(rangeLags, xcf)
        baseline.set_color('black')
        plt.xlabel('Lag')
        plt.ylabel('Amplitude')
        condInt = 2/np.sqrt(N) * np.ones(len(rangeLags))
        maxRange = max(max(np.abs(xcf)), 2/np.sqrt(N))*1.1
        plt.plot(rangeLags, condInt, 'red', linestyle='--')
        plt.plot(rangeLags, -condInt, 'red', linestyle='--')
        plt.axis([rangeLags[0], rangeLags[-1], -maxRange, maxRange])
        plt.axvline(0, c='black', ls='-', lw=0.7)
        plt.title(f'Normalized Cross Correlation of {titleStr}') if titleStr else plt.title('Normalized Cross Correlation')
        plt.grid()
        plt.show()
        return
    
    return xcf

def normplot(X, titleStr=None, return_val=False):
    """
    Creates a normal probability plot comparing the distribution of the data in X to the normal distribution.

    Parameters:
    - X (list/array-like): Data for which the normal probability plot is to be created.
    - titleStr (str, optional): A string to be appended in the title of the plot. Defaults to an empty string.
    - return_val (bool, optional): If set to True, returns the ordered values of the data. Defaults to False.

    Returns:
    - list/array-like: If return_val is True, returns the ordered values of the data. Otherwise, nothing is returned.
    """

    (quantiles, values), (slope, intercept, r) = scipy.stats.probplot(X, dist='norm')

    plt.plot(values, quantiles,'xb')
    plt.plot(quantiles * slope + intercept, quantiles, 'r')

    #define ticks
    ticks_perc=[0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.99]
    #transfrom them from precentile to cumulative density
    ticks_quan=[scipy.stats.norm.ppf(i) for i in ticks_perc]
    plt.yticks(ticks_quan,ticks_perc)

    plt.title(f'Normal probability plot ({titleStr})') if titleStr else plt.title(f'Normal probability plot')
    plt.grid()
    plt.show()
    if return_val:
        return values

def resid(fitted_model, y, useFilter=True, plotACF=False, noLags='auto'):
    """
    ** REPLACED BY RESID ATTRIBUTE IN PEM**
    Computes the residuals for a given model and observed data.

    Parameters:
    - model (object): An object with attributes 'polynomial_ar' and 'polynomial_ma' representing the AR and MA components, respectively, and 'resid' for the residuals.
    - y (array-like): The observed data.
    - useFilter (bool, optional): If True, creates residuals using the inverted filter with the AR and MA polynomials. If False, directly uses the 'resid' attribute from the model. Default is True.
    - plotACF (bool, optional): If True, plots the Auto-Correlation Function (ACF) of the residuals. Default is False.
    - noLags (int, optional): Number of lags to be used in the ACF plot. Default is 0.

    Returns:
    - res (array-like): The residuals computed for the model and data.
    """
    if useFilter:
        res = scipy.signal.lfilter(fitted_model.polynomial_ar, fitted_model.polynomial_ma, y)
    else:
        res = fitted_model.resid[len(fitted_model.polynomial_ar)-1:]
    
    if plotACF:
        acf(res,maxOrd=noLags,plotIt=True)
        plt.title('Residual ACF')
        plt.show()

    return res

def box_cox(y, plotIt=True, titleStr=None, lamRange=[-2,2], noVals=100, transform=True):
    """
    Perform a Box-Cox transformation on a dataset.

    Parameters:
    - y (iterable): Data to be transformed. Must be a 1-D array-like structure containing numeric values.
    - plotIt (bool): If True, plots the log-likelihood for each lambda value within 'lamRange'. Defaults to True.
    - titleStr (str, optional): Title of the plot. If None, a default title is used. Defaults to None.
    - lamRange (list): Two-element list specifying the range of lambda values to consider for the transformation. Defaults to [-2, 2].
    - noVals (int): Specifies the number of values to consider between 'lamRange[0]' and 'lamRange[1]'. Affects the resolution of the plot. 
        Defaults to 100.
    - transform (bool): If True, the function suggests a possible transformation based on the lambda value. 
        If False, returns lambda and offset values. Defaults to True.

    Returns:
    - tuple: A tuple containing the lambda value and offset used for transformation, returned only if 'transform' is False. 
        Otherwise, no explicit return value.
    """
    y = y.copy()
    lamRange = np.linspace(lamRange[0], lamRange[1], noVals)

    offsetValue = 0
    if np.min(y) <= 0:
        offsetValue = 1 - np.min(y)
        y += offsetValue

    N = len(y)
    bct = np.zeros(len(lamRange))
    for k, lambda_ in enumerate(lamRange):
        if lambda_ == 0:
            z = np.log(y)
        else:
            z = (y**lambda_ - 1) / lambda_

        bct[k] = -(N/2) * np.log(np.std(z)**2) + (lambda_-1) * np.sum(np.log(y))

    # maxLam = lamRange[np.argmax(bct)]
    maxLam = scipy.stats.boxcox(y)[1] # More exact

    if plotIt:
        plt.plot(lamRange, bct)
        plt.xlabel('Lambda')
        plt.ylabel('Log-Likelihood')
        plt.title(f'Box-Cox Normality Plot ({titleStr})') if titleStr else plt.title('Box-Cox Normality Plot')
        plt.grid()
        plt.show()

    if transform:
        print(f'Max Lambda = {maxLam}.')
        transforms = {-3: 'y^-3', 
                      -2: 'y^-2', 
                      -1: 'y^-1', 
                      -0.5: 'y^-0.5', 
                      0: 'ln(y)', 
                      0.5: 'y^0.5', 
                      1: 'y', 
                      2: 'y^2', 
                      3: 'y^3'}
        t = min(list(transforms.keys()), key=lambda x: abs(x - maxLam))
        if t==1:
            print('Transformation is likely not needed.')
        else:
            print(f'{transforms[t]} could be an appropriate transformation.')
        
        return
    return maxLam, offsetValue

def pzmap(b,a, return_val=False):
    """Plot the complex z-plane given a transfer function.
    Copyright (c) 2011 Christopher Felton
    https://www.dsprelated.com/showcode/244.php
    """
    b = np.array(b)
    a = np.array(a)

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    max_value = max(abs(np.max(p.real)), abs(np.max(p.imag)), abs(np.max(z.real)), abs(np.max(z.imag)))
    r = max_value + 0.5
    plt.axis('scaled')
    plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    plt.show()
    
    if return_val:
        return z, p, k

def covMvect(y, meanD=None, fb=False):
    """
    Computes the unbiased covariance matrix estimate of the data,
    which is assumed to consist of N row vectors, formed as an N x m matrix. 
    The data is assumed to have a 1 x m mean vector meanD; if not given, it is 
    estimated. If fb is set, forward-backward averaging is used.
    """
    data = np.atleast_2d(y)
    m,N = data.shape
    if meanD is None:
        meanD = np.mean(data, axis=1).reshape(-1,1)
    else:
        
        m1,n1 = meanD.shape
        # Check dimensionality
        if n1 == m and m1 == 1:  # If flipped, transpose to row vector.
            meanD = meanD.T
            m1, n1 = n1, m1
        if m1 != m:
            raise ValueError("Incompatible dimensions for mean vector and data.")
    
    # Compensate for mean value
    data = data - np.ones((m,N)) * meanD
    
    # Form forward-only covariance matrix
    R = data @ data.T / (N-1)
    
    if fb:
        H = np.fliplr(np.eye(m))  # Exchange matrix
        R = (R + H @ R.T @ H) / 2  # Form forward-backward estimate
    
    return R, meanD

def mat2pd(file_path, columns=None):
    """
    Reads a .mat file and returns a Pandas DataFrame.

    Parameters:
    - file_path (str): The path to the .mat file.

    Returns:
    - df (pd.DataFrame): The DataFrame containing the data from the .mat file.

    Note:
    Assumes there is only one key in the mat file.
    """
    data = scipy.io.loadmat(file_path)
    key = [key for key in data.keys() if not key.startswith("__")][0]
    array = data[key]

    if columns is None:
        columns = [key] if array.shape[1]==1 else [f'col{n+1}' for n in range(array.shape[1])]

    return pd.DataFrame(array, columns=columns)
    
def mat2np(file_path):
    """
    Reads a .mat file and returns a numpy array.

    Parameters:
    - file_path (str): The path to the .mat file.

    Returns:
    - np.array: The array containing the data from the .mat file.

    Note:
    Assumes there is only one key in the mat file.
    """
    data = scipy.io.loadmat(file_path)
    key = [key for key in data.keys() if not key.startswith("__")][0]
    array = np.array(data[key])
    if len(array.shape)>1:
        if array.shape[0]==1 or array.shape[1]==1:
            array = array.ravel()

    return array
  
def mfile2pd(filename):
    """
    Reads a m file by Andreas and returns a pandas dataframe.
    Works pretty bad.
    """

    # Open and read the .m file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Remove comments and empty lines
    lines = [line for line in lines if not line.strip().startswith('%') and line.strip()]

    # Extract matrix data by searching between square brackets
    matrix_str = ''.join(lines)
    matrix_match = re.search(r'\[([\s\S]*?)\]', matrix_str)  # Non-greedy match to capture only the matrix
    if matrix_match:
        matrix_str = matrix_match.group(1).strip()
    else:
        raise ValueError("No matrix data found in the file.")

    # Split the matrix string into lines and process each line
    matrix_lines = matrix_str.split("\n")
    matrix = [list(map(float, row.strip().split())) for row in matrix_lines if row.strip()]

    # Convert to pandas DataFrame
    df = pd.DataFrame(matrix)
    
    return df