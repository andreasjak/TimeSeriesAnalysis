"""
Extended Sample Autocorrelation Function (ESACF)

This module provides ESACF estimation for ARMA model order identification
following the Tsay & Tiao (1984) algorithm. It is a direct translation of
the MATLAB functions esacf.m and arIterEsacf.m used in the course.
"""

import numpy as np
from scipy import signal


def ar_iter_esacf(x, j, k):
    """
    Compute the iterative AR estimates used in the ESACF estimate.
    Direct translation of MATLAB arIterEsacf function.

    Returns the AR polynomial A = [1, a_1, ..., a_k] of the j:th iteration.
    """
    if j < 0:
        raise ValueError('arIterEsacf: Illegal function call.')
    elif j == 0:
        A = ar_covariance_method(x, k)
    else:
        A_k = ar_iter_esacf(x, j-1, k)
        A_k1 = ar_iter_esacf(x, j-1, k+1)
        A = np.zeros(k + 1)
        A[0] = 1.0
        for ell in range(1, k+1):  # ell = 1 to k in MATLAB
            A[ell] = A_k1[ell] - A_k[ell-1] * A_k1[k+1] / A_k[k]
    return A


def ar_covariance_method(x, order):
    """
    Estimate an AR(order) polynomial using the covariance method, as MATLAB's arcov.

    The coefficients minimize the forward prediction error sum over t = order, ..., N-1.

    Returns the AR polynomial A = [1, a_1, ..., a_order].
    """
    x = np.asarray(x, dtype=float).flatten()
    N = len(x)
    # Row t holds [x_{t-1}, ..., x_{t-order}] for t = order, ..., N-1.
    X = np.column_stack([x[order-i:N-i] for i in range(1, order+1)])
    a, *_ = np.linalg.lstsq(X, -x[order:], rcond=None)
    return np.concatenate(([1.0], a))


def esacf(x, pMax, qMax):
    """
    Compute the ESACF estimate for ARMA model order identification.
    Direct translation of MATLAB esacf function following Tsay & Tiao (1984).

    Parameters:
    - x (array-like): Data.
    - pMax (int): Maximum AR order.
    - qMax (int): Maximum MA order.

    Returns:
    - esacfM (ndarray): The (pMax+1) x qMax ESACF matrix. Row p holds the AR order p = 0, ..., pMax,
      and column q the MA order q = 0, ..., qMax-1.
    - esacfX (ndarray): True where |esacfM| is larger than condInt.
    - condInt (float): The approximate 95% confidence bound.
    """
    x = np.asarray(x, dtype=float).flatten()
    x = x - np.mean(x)
    N = len(x)
    esacfM = np.zeros((pMax + 1, qMax))

    def normalized_acf(z):
        """As MATLAB's r = xcorr(z, qMax, 'biased'); r = r/max(r); r = r(qMax+2:end)."""
        n = len(z)
        r = np.array([np.sum(z[:n-k] * z[k:]) / n for k in range(qMax + 1)])
        return r[1:] / r[0]

    esacfM[0, :] = normalized_acf(x)
    for k in range(1, pMax + 1):
        for j in range(qMax):
            A = ar_iter_esacf(x, j + 1, k)
            z = signal.lfilter(A, 1, x)
            z = z[len(A)-1:]
            esacfM[k, j] = normalized_acf(z)[j]

    condInt = 2.0 / np.sqrt(N - pMax - qMax)
    esacfX = np.abs(esacfM) > condInt
    return esacfM, esacfX, condInt
