#Marcus Lindell & Casper Schwerin
from statsmodels.tsa.stattools import acf, pacf 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm
from tools.niceplot import plot_colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import boxcox_normplot, boxcox

#tools for plotting acf/pacf
def acf_pacf(data, lags=20, signLvl=0.05):

    fig, ax = plt.subplots(nrows=2, ncols=1, facecolor="#F0F0F0")

    signScale = norm.ppf(1 - signLvl/2)
    condInt = norm.ppf(1 - signLvl/2)/np.sqrt(len(data)) # change 1000 to len(data)


    plot_acf(data, lags=lags, alpha=condInt, ax=ax[0], title='ACF of Modelling data', bartlett_confint=False)
    ax[0] = plot_colors(ax[0])


    plot_pacf(data, lags=lags, alpha=condInt, ax=ax[1], title='PACF of Modelling data', method='ywm')
    ax[1] = plot_colors(ax[1])

    plt.tight_layout()

def boxcox_plot(data):
    fig_boxcox = plt.figure()
    ax_boxcox = fig_boxcox.add_subplot(111)
    lambdas, ppcc = boxcox_normplot(data, -6, 6, plot=ax_boxcox)
    maxlog = lambdas[np.argmax(ppcc)]
    ax_boxcox.axvline(maxlog, color='r')
    return maxlog