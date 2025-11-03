#Marcus Lindell & Casper Schwerin
import numpy as np
def fisherTest(data, alpha=0.05):
    #this test checks IF there is a significant frequency present. 1 = yes, 0 = no

    N= len(data)
    X = np.divide(np.power(np.fft.fft(data),2), N)
    X = np.abs(X)

    N2 = np.floor(N/2-0.5)
    N2 = N2.astype(int)

    X2 = X[0:N2+1]

    T= np.max(X2)/np.sum(X2)

    g = 1- np.power((alpha/N2), 1/(N2-1))

    if(T>g):
        result = 1
    else:
        result = 0


    return result