#Marcus Lindell & Casper Schwerin
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np

#plot fft
def fft_plot(data):

    f, Pxx = signal.periodogram(data)
    plt.plot(f, Pxx)
    plt.xlabel("Frequency")

    fig, ax = plt.subplots()
    ax.loglog(f, Pxx)
    ax.set_xlabel('Frequency')

    def invert(x):
        # 1/x with special treatment of x == 0
        x = np.array(x).astype(float)
        near_zero = np.isclose(x, 0)
        x[near_zero] = np.inf
        x[~near_zero] = 1 / x[~near_zero]
        return x

    # the inverse of 1/x is itself
    secax = ax.secondary_xaxis('top', functions=(invert, invert))
    secax.set_xlabel('Period')
    plt.show()
    