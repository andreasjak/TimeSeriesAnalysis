"""
Multivariate Time Series Analysis Functions

This module contains functions for analyzing multivariate time series,
including Vector Autoregressive (VAR) models, multivariate ACF/PACF,
and related statistical tests.

Reference:
    "An Introduction to Time Series Modeling", 4th ed, by Andreas Jakobsson
    Studentlitteratur, 2021
"""

import numpy as np
from scipy.stats import chi2
from tsa_lth.analysis import xcorr


def corrM(Y, L):
    """
    Estimate auto-covariance and auto-correlation for multivariate process.
    
    Computes the sample auto-covariance and auto-correlation functions
    for a multivariate time series using the biased estimator.
    
    Parameters
    ----------
    Y : array_like
        m x N matrix where m is the dimension and N is the number of samples,
        or N x m matrix (will be transposed automatically)
    L : int
        Maximum lag to compute
    
    Returns
    -------
    Ry : list of ndarray
        List of m x m covariance matrices for lags 0 to L
    rhoY : list of ndarray
        List of m x m correlation matrices for lags 0 to L
        
    Notes
    -----
    The function uses the biased estimator, dividing by N rather than (N-k)
    for lag k. The correlation matrices are normalized versions of the
    covariance matrices.
    
    Examples
    --------
    >>> Y = np.random.randn(2, 1000)  # 2-dimensional process, 1000 samples
    >>> Ry, rhoY = corrM(Y, 10)  # Compute for lags 0-10
    >>> print(f"Covariance at lag 0 shape: {Ry[0].shape}")
    Covariance at lag 0 shape: (2, 2)
    """
    Y = np.atleast_2d(Y)
    m, N = Y.shape
    if N < m:
        Y = Y.T
        m, N = Y.shape
    
    mY = np.mean(Y, axis=1, keepdims=True)
    Ry = []
    
    for k in range(L + 1):
        R_k = np.zeros((m, m))
        for p in range(k, N):
            R_k += (Y[:, p:p+1] - mY) @ (Y[:, p-k:p-k+1] - mY).T
        R_k /= N
        Ry.append(R_k)
    
    # Normalize to get correlation
    D = np.diag(1.0 / np.sqrt(np.diag(Ry[0])))
    rhoY = [D @ R @ D for R in Ry]
    
    return Ry, rhoY


def pacfM(Y, L):
    """
    Estimate partial auto-correlation for multivariate process.
    
    Computes the multivariate partial auto-correlation function (PACF)
    using the Levinson-Durbin recursion with forward-backward averaging.
    
    Parameters
    ----------
    Y : array_like
        m x N matrix where m is the dimension and N is the number of samples,
        or N x m matrix (will be transposed automatically)
    L : int
        Maximum lag to compute
    
    Returns
    -------
    S : list of ndarray
        List of m x m partial correlation matrices for lags 1 to L
        (Note: S[0] corresponds to lag 1, S[1] to lag 2, etc.)
    Ry : list of ndarray
        List of m x m covariance matrices for lags 0 to L (from corrM)
        
    Notes
    -----
    The function uses the multivariate Levinson-Durbin algorithm to
    recursively compute the partial correlation matrices. Each matrix
    S[k-1] represents the partial correlation at lag k.
    
    The algorithm performs forward-backward estimation to obtain more
    stable estimates, particularly for shorter time series.
    
    Examples
    --------
    >>> Y = np.random.randn(2, 1000)
    >>> S, Ry = pacfM(Y, 10)
    >>> print(f"Number of PACF matrices: {len(S)}")
    Number of PACF matrices: 10
    >>> print(f"PACF at lag 1 shape: {S[0].shape}")
    PACF at lag 1 shape: (2, 2)
    """
    Ry, _ = corrM(Y, L)
    
    # Initialize
    Vu = [None] * (L + 1)
    Vv = [None] * (L + 1)
    Vvu = [None] * (L + 1)
    alp = [[None] * L for _ in range(L)]
    bet = [[None] * L for _ in range(L)]
    S = []
    
    # First iteration (s=1)
    Vu[1] = Ry[0].copy()
    Vv[1] = Vu[1].copy()
    Vvu[1] = Ry[1].copy()
    
    tmp = np.linalg.inv(Ry[0])
    alp[0][0] = Ry[1].T @ tmp
    bet[0][0] = Ry[1] @ tmp
    
    Dv = np.diag(1.0 / np.sqrt(np.diag(Vv[1])))
    Du = np.diag(1.0 / np.sqrt(np.diag(Vu[1])))
    S.append(Dv @ Vvu[1].T @ Du)
    
    # Iterations s=2 to L
    for s in range(2, L + 1):
        Vu[s] = Ry[0].copy()
        Vv[s] = Ry[0].copy()
        Vvu[s] = Ry[s].copy()
        
        for k in range(1, s):
            Vu[s] -= alp[s-2][k-1] @ Ry[k]
            Vv[s] -= bet[s-2][k-1] @ Ry[k].T
            Vvu[s] -= Ry[s-k] @ alp[s-2][k-1].T
        
        alp[s-1][s-1] = Vvu[s].T @ np.linalg.inv(Vv[s])
        bet[s-1][s-1] = Vvu[s] @ np.linalg.inv(Vu[s])
        
        for k in range(1, s):
            alp[s-1][k-1] = alp[s-2][k-1] - alp[s-1][s-1] @ bet[s-2][s-k-1]
            bet[s-1][k-1] = bet[s-2][k-1] - bet[s-1][s-1] @ alp[s-2][s-k-1]
        
        Dv = np.diag(1.0 / np.sqrt(np.diag(Vv[s])))
        Du = np.diag(1.0 / np.sqrt(np.diag(Vu[s])))
        S.append(Dv @ Vvu[s] @ Du)
    
    return S, Ry


def lsVAR(y, p):
    """
    Estimate VAR(p) model using least squares.
    
    Estimates the parameters of a Vector Autoregressive model of order p
    using the least squares method. Returns the AR coefficient matrices,
    residual covariance matrix, and prediction residuals.
    
    Parameters
    ----------
    y : array_like
        m x N matrix where m is the dimension and N is the number of samples,
        or N x m matrix (will be transposed automatically)
    p : int
        AR order (number of lags). Use p=0 for white noise model.
    
    Returns
    -------
    thEst : ndarray
        m x (m*p) matrix of AR coefficients, where columns [0:m] contain
        coefficients for lag 1, columns [m:2m] for lag 2, etc.
    sigEst : ndarray
        m x m residual covariance matrix
    resEst : ndarray
        (N-p) x m matrix of residuals (prediction errors)
        
    Notes
    -----
    For a VAR(p) model:
        y(t) = -A1*y(t-1) - A2*y(t-2) - ... - Ap*y(t-p) + e(t)
    
    The coefficient matrices are stacked horizontally in thEst as:
        thEst = [A1 A2 ... Ap]
    
    Special case: When p=0, returns white noise model where all observations
    are treated as residuals and covariance is computed from all data.
    
    Examples
    --------
    >>> Y = np.random.randn(2, 1000)
    >>> thEst, sigEst, resEst = lsVAR(Y, 2)  # Fit VAR(2)
    >>> print(f"Coefficient matrix shape: {thEst.shape}")
    Coefficient matrix shape: (2, 4)
    >>> print(f"A1 = {thEst[:, 0:2]}")  # First lag coefficients
    >>> print(f"Residual covariance:\n{sigEst}")
    """
    y = np.atleast_2d(y)
    m, N = y.shape
    if m > N:
        y = y.T
        m, N = y.shape
    
    # Special case for p=0 (white noise model)
    if p == 0:
        thEst = np.zeros((m, 0))
        sigEst = np.cov(y)
        resEst = y.T  # All data are residuals
        return thEst, sigEst, resEst
    
    # Build Y and X matrices
    Y = y[:, p:N].T  # (N-p) x m
    X = np.zeros((N - p, m * p))
    
    for k in range(p, N):
        # Stack [y(k-1), y(k-2), ..., y(k-p)] into row k-p
        for j in range(p):
            X[k-p, j*m:(j+1)*m] = -y[:, k-1-j]
    
    # Least squares estimation
    thEst = np.linalg.inv(X.T @ X) @ X.T @ Y
    resEst = Y - X @ thEst
    sigEst = resEst.T @ resEst / (N - p)
    
    thEst = thEst.T  # m x (m*p)
    
    return thEst, sigEst, resEst


def lbp_test_multivariate(data, K=20, alpha=0.05):
    """
    Ljung-Box-Pierce test for multivariate time series.
    
    Computes the multivariate version of the Ljung-Box-Pierce whiteness
    test to assess whether a multivariate time series or set of residuals
    exhibits significant autocorrelation.
    
    Parameters
    ----------
    data : array_like
        Residuals as (N-p) x m or m x N matrix, where m is the dimension
        and N is the number of samples
    K : int, optional
        Number of lags to include in the test (default: 20)
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    deemedWhite : bool
        True if residuals are deemed white (no significant autocorrelation)
    Q : float
        Test statistic value
    chiV : float
        Chi-squared critical value at significance level alpha
        
    Notes
    -----
    The null hypothesis is that the data is white noise (no autocorrelation).
    If Q < chiV, we fail to reject the null hypothesis and deem the data white.
    
    For univariate data (m=1), uses the standard univariate LBP test.
    For multivariate data (m>1), uses the multivariate formulation based
    on the trace of products of cross-correlation matrices.
    
    The test statistic approximately follows a chi-squared distribution
    with K*m^2 degrees of freedom under the null hypothesis.
    
    Examples
    --------
    >>> residuals = np.random.randn(1000, 2)  # White noise residuals
    >>> deemedWhite, Q, chiV = lbp_test_multivariate(residuals, K=20)
    >>> print(f"White? {deemedWhite}, Q={Q:.2f}, threshold={chiV:.2f}")
    """
    data = np.atleast_2d(data)
    
    # Handle both orientations
    if data.shape[0] > data.shape[1]:
        data = data.T  # m x N
    
    m, N = data.shape
    
    # If univariate, use standard test
    if m == 1:
        r = xcorr(data[0], maxlag=K, biased=True)[1]
        r = r / np.max(r)
        r = r[K+1:2*K+1]
        Q = N * (N + 2) * np.sum(r**2 / np.arange(N-1, N-K-1, -1))
    else:
        # Multivariate LBP test
        Q = 0
        C0_inv = np.linalg.inv(np.cov(data))
        
        for k in range(1, K + 1):
            Ck = np.zeros((m, m))
            for t in range(k, N):
                Ck += data[:, t:t+1] @ data[:, t-k:t-k+1].T
            Ck /= N
            
            Q += np.trace(Ck.T @ C0_inv @ Ck @ C0_inv) / (N - k)
        
        Q *= N**2
    
    chiV = chi2.ppf(1 - alpha, K * m**2)
    deemedWhite = Q < chiV
    
    return deemedWhite, Q, chiV


def var_select_order(Y, max_order=10, criterion='AIC'):
    """
    Select optimal VAR model order using information criteria.
    
    Estimates VAR models from order 0 to max_order and computes information
    criteria (AIC, BIC, FPE) to help select the optimal model order.
    
    Parameters
    ----------
    Y : array_like
        m x N matrix where m is the dimension and N is the number of samples
    max_order : int, optional
        Maximum order to consider (default: 10)
    criterion : str, optional
        Selection criterion: 'AIC', 'BIC', or 'FPE' (default: 'AIC')
    
    Returns
    -------
    optimal_order : int
        Optimal model order according to the specified criterion
    criteria : dict
        Dictionary with 'AIC', 'BIC', and 'FPE' arrays for orders 0 to max_order
    models : dict
        Dictionary with 'theta', 'sigma', and 'residuals' lists for all estimated models
        
    Examples
    --------
    >>> Y = np.random.randn(2, 1000)
    >>> order, criteria, models = var_select_order(Y, max_order=6)
    >>> print(f"Optimal order: {order}")
    >>> print(f"AIC values: {criteria['AIC']}")
    """
    Y = np.atleast_2d(Y)
    m, N = Y.shape
    
    AIC = np.zeros(max_order + 1)
    BIC = np.zeros(max_order + 1)
    FPE = np.zeros(max_order + 1)
    
    thetas = []
    sigmas = []
    residuals = []
    
    for k in range(max_order + 1):
        th, sig, res = lsVAR(Y, k)
        thetas.append(th)
        sigmas.append(sig)
        residuals.append(res)
        
        AIC[k] = N * np.log(np.linalg.det(sig)) + 2*k*m**2
        BIC[k] = N * np.log(np.linalg.det(sig)) + k*m**2 * np.log(N)
        FPE[k] = ((N + m*k + 1) / (N - m*k - 1))**m * np.linalg.det(sig)
    
    criteria = {'AIC': AIC, 'BIC': BIC, 'FPE': FPE}
    models = {'theta': thetas, 'sigma': sigmas, 'residuals': residuals}
    
    if criterion == 'AIC':
        optimal_order = np.argmin(AIC)
    elif criterion == 'BIC':
        optimal_order = np.argmin(BIC)
    elif criterion == 'FPE':
        optimal_order = np.argmin(FPE)
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Use 'AIC', 'BIC', or 'FPE'.")
    
    return optimal_order, criteria, models


def likelihood_ratio_test(Y, max_order=10, alpha=0.05):
    """
    Likelihood ratio test for VAR model order selection.
    
    Performs sequential likelihood ratio tests to determine the appropriate
    VAR model order. Tests the null hypothesis that order p-1 is sufficient
    against the alternative that order p is needed.
    
    Parameters
    ----------
    Y : array_like
        m x N matrix where m is the dimension and N is the number of samples
    max_order : int, optional
        Maximum order to test (default: 10)
    alpha : float, optional
        Significance level (default: 0.05)
    
    Returns
    -------
    test_statistics : ndarray
        Array of likelihood ratio test statistics for orders 1 to max_order
    threshold : float
        Chi-squared critical value
    significant : ndarray
        Boolean array indicating which order increases are significant
    suggested_order : int
        Last order where the test statistic exceeds the threshold
        
    Notes
    -----
    The test statistic is:
        M = -(N - p - p*m - 0.5) * log(|Sigma_p| / |Sigma_{p-1}|)
    
    which follows a chi-squared distribution with m^2 degrees of freedom
    under the null hypothesis.
    
    Examples
    --------
    >>> Y = np.random.randn(2, 1000)
    >>> M, chiV, sig, order = likelihood_ratio_test(Y, max_order=6)
    >>> print(f"Suggested order: {order}")
    """
    Y = np.atleast_2d(Y)
    m, N = Y.shape
    
    # Estimate models for all orders
    sigmas = []
    for k in range(max_order + 1):
        _, sig, _ = lsVAR(Y, k)
        sigmas.append(sig)
    
    # Compute test statistics
    M = np.zeros(max_order)
    for p in range(1, max_order + 1):
        M[p-1] = -(N - p - p*m - 0.5) * np.log(np.linalg.det(sigmas[p]) / np.linalg.det(sigmas[p-1]))
    
    chiV = chi2.ppf(1 - alpha, m**2)
    significant = M > chiV
    
    # Find last significant order
    if np.any(significant):
        suggested_order = np.where(significant)[0][-1] + 1
    else:
        suggested_order = 0
    
    return M, chiV, significant, suggested_order


# Convenience alias for backward compatibility
lbp_test = lbp_test_multivariate



