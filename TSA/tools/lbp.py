#Marcus Lindell & Casper Schwerin
import numpy as np
from scipy.stats.distributions import chi2
from statsmodels.tsa.stattools import acf

#ljung box pierce test
def lbp_test(data, k=24 , alpha=0.05):
    N = len(data)

    r = acf(data, nlags=k)

    Q = N*(N+2)*np.sum(np.divide(np.power(r[1:],2), np.subtract(N, np.array(range(0,k)).T)))

    chiV = chi2.ppf(1-alpha, df=k)
    deemedWhite = 1 if Q < chiV else 0

    Q = round(Q, 2)
    chiV = round(chiV, 2)

    return (deemedWhite, Q, chiV)