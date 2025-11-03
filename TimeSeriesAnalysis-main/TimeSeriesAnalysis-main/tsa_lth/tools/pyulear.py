
import numpy as np
from scipy.signal import freqz
from statsmodels.regression.linear_model import yule_walker


def pyulear(y, order, nfft, option="centered", fs=1.0):
    """
    Estimate the power spectral density using the Yule–Walker AR method.

    Parameters
    ----------
    y : array_like
        Input signal (1D).
    order : int
        AR model order (e.g. 20).
    nfft : int
        Number of FFT points (zero-padding for frequency resolution).
    option : str, optional
        'centered' (default) -> two-sided, centered spectrum
        'onesided'          -> one-sided PSD (0..fs/2)
    fs : float, optional
        Sampling frequency in Hz (default=1.0, normalized units).

    Returns
    -------
    Pxx : ndarray
        Power spectral density estimate.
    f : ndarray
        Frequency axis in Hz if fs given, else cycles/sample.
    """
    y = np.asarray(y)
    y = y - y.mean()  # remove mean, like MATLAB

    # Estimate AR coefficients and noise variance
    phi, sigma = yule_walker(y, order=order, method="mle", demean=False)
    a = np.r_[1.0, -phi]  # AR denominator polynomial
    sigma2 = sigma**2

    # Frequency response (full circle, [0, 2π))
    w, H = freqz([np.sqrt(sigma2)], a=a, worN=nfft, whole=True)
    Pxx = np.abs(H) ** 2

    if option.lower() == "centered":
        # Two-sided, center zero frequency
        f = (w / (2 * np.pi)) * fs - fs / 2.0
        Pxx = np.fft.fftshift(Pxx)
    elif option.lower() == "onesided":
        # One-sided [0, fs/2]
        w = w[: nfft // 2 + 1]
        Pxx = Pxx[: nfft // 2 + 1]
        f = (w / (2 * np.pi)) * fs
        if np.isrealobj(y):
            # scale to preserve power (like MATLAB)
            Pxx[1:-1] *= 2.0
    else:
        raise ValueError("option must be 'centered' or 'onesided'")

    if fs != 1.0:
        Pxx = Pxx / fs

    return Pxx, f

