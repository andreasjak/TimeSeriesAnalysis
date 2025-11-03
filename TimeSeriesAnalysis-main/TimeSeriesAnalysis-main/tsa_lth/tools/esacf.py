"""
Extended Sample Autocorrelation Function (ESACF)

This module provides ESACF estimation for ARMA model order identification
following the Tsay & Tiao (1984) algorithm.
"""

import numpy as np
from scipy import signal
from scipy.signal import correlate


def ar_iter_esacf(x, j, k):
    """
    Compute the iterative AR estimates used in the ESACF estimate.
    Direct translation of MATLAB arIterEsacf function.
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
            if (len(A_k) > ell and len(A_k1) > ell and 
                len(A_k1) > k+1 and len(A_k) > k and A_k[k] != 0):
                A[ell] = A_k1[ell] - A_k[ell-1] * A_k1[k+1] / A_k[k]
    
    return A


def ar_covariance_method(x, order):
    
    from scipy.linalg import solve, LinAlgError, lstsq
    
    x = np.asarray(x).flatten()
    x = x - np.mean(x)  # Center the data like MATLAB
    N = len(x)
    
    if order >= N or order <= 0:
        A = np.zeros(order + 1)
        A[0] = 1.0
        return A
    
    try:
        # Forward-backward method like MATLAB arcov
        # Create data matrix for least squares
        n_samples = N - order
        X = np.zeros((2 * n_samples, order))
        y = np.zeros(2 * n_samples)
        
        # Forward equations: x[n] = -a[1]*x[n-1] - ... - a[p]*x[n-p]
        for i in range(n_samples):
            X[i, :] = x[i:i + order][::-1]  # Reverse for AR convention
            y[i] = -x[i + order]
        
        # Backward equations: x[n] = -a[1]*x[n+1] - ... - a[p]*x[n+p]  
        for i in range(n_samples):
            idx = i + n_samples
            X[idx, :] = x[N - 1 - i - order:N - 1 - i]  # Forward for backward
            y[idx] = -x[N - 1 - i - order]
        
        # Solve least squares problem
        ar_coeffs, _, _, _ = lstsq(X, y, rcond=None)
        
        # Construct result
        A = np.zeros(order + 1)
        A[0] = 1.0
        A[1:] = ar_coeffs
        
    except:
        A = np.zeros(order + 1)
        A[0] = 1.0
    
    return A


def esacf(x, pMax, qMax):
    """
    Compute the ESACF estimate for ARMA model order identification.
    
    Direct translation of MATLAB esacf function following Tsay & Tiao (1984).
    """
    
   
    x = np.asarray(x).flatten()
    x = x - np.mean(x)  # Center the data
    N = len(x)
    esacfM = np.zeros((pMax, qMax))  # 

    def matlab_xcorr_biased(x, maxlags):
        """Replicate MATLAB's xcorr(x, maxlags, 'biased') exactly"""
        x = x - np.mean(x)
        n = len(x)
        
        
        
        r_vals = []
        for k in range(1, maxlags + 1):
            if k < n:
                r_vals.append(np.sum(x[:-k] * x[k:]) / n)  # biased correlation
            else:
                r_vals.append(0.0)
        
        return np.array(r_vals)
    
   
    r = matlab_xcorr_biased(x, qMax)
    max_r = np.max(r)  
    if max_r != 0:  
        r = r / max_r 
    esacfM[0, :] = r 
    
    
    for k in range(1, pMax): 
        for j in range(qMax): 
            try:
                
                A = ar_iter_esacf(x, j + 1, k)
                
                
                z = signal.lfilter(A, 1, x)
                z = z[len(A)-1:]  
                
                
                rz = matlab_xcorr_biased(z, qMax)
                max_rz = np.max(rz)  
                if max_rz != 0:  
                    rz = rz / max_rz  
                
                
                if j < len(rz):
                    esacfM[k, j] = rz[j] 
                else:
                    esacfM[k, j] = 0.0
                    
            except:
                esacfM[k, j] = 0.0
    
    
    condInt = 2.0 / np.sqrt(len(x) - pMax - qMax + 1)
    esacfX = np.abs(esacfM) > condInt

    print("Warning: ESACF implemented for demonstration and is not fully working, do not trust the results.")
    return esacfM, esacfX, condInt