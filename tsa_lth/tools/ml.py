#Marcus Lindell & Casper Schwerin
import numpy as np
from scipy.stats.distributions import chi2
from statsmodels.tsa.stattools import acf

#mcleod-li test
def ml_test(data, k=24, alpha=0.05):
    N = len(data)

    r = acf(np.power(data,2) - np.mean(np.power(data,2)), nlags=k)

    Q = N*(N+2)*np.sum(np.divide(np.power(r[1:],2), np.subtract(N, np.array(range(0,k)).T)))

    chiV = chi2.ppf(1-alpha, df=k)
    deemedWhite = 1 if Q < chiV else 0

    Q = round(Q, 2)
    chiV = round(chiV, 2)

    return (deemedWhite, Q, chiV)