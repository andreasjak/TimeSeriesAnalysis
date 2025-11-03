#Marcus Lindell & Casper Schwerin
import numpy as np
from scipy.stats import norm

#sign test
def sign_test(x, alpha=0.05):
    data = x.copy()
    N = len(data)

    for i in range(N):

        if data[i] > 0:
            data[i] = 1

        else:
            data[i] = 0
    
    changes = 0

    for i in range(N-1):
        if data[i] - data[i+1] != 0:
            changes += 1

    nRatio = changes/(N-1)

    if (1 - alpha > 1) or (1 - alpha < 0):
        print('Poor choice of confidence level')
    
    normConfLev = norm.ppf((1 - alpha/2), 0, 1)
    confInt = ((N-1)/2 + normConfLev*np.sqrt((N-1)/4) * np.array([-1, 1])) / (N - 1)

    nRatio = round(nRatio, 2)
    for i in range(2):
        confInt[i] = round(confInt[i], 2)

    return (nRatio, confInt)
