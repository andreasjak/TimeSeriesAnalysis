#Marcus Lindell & Casper Schwerin
import numpy as np
from scipy.stats.distributions import chi2
from statsmodels.tsa.stattools import pacf

#monti test
def monti_test(data, k=24 , alpha=0.05): #data is a DataFrame? or just array-like?
    N = len(data)

    r = pacf(data, k, method="ldb")

    Q = N*(N+2)*np.sum(np.divide(np.power(r[1:], 2), np.subtract(N, np.array(range(0,k)).T)))

    chiV = chi2.ppf(1-alpha, df=k)
    deemedWhite = 1 if Q < chiV else 0

    Q = round(Q, 2)
    chiV = round(chiV, 2)

    return (deemedWhite, Q, chiV)