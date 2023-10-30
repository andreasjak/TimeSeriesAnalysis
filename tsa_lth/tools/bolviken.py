#Marcus Lindell & Casper Schwerin
import numpy as np
def bolvikenTest(data, na=8, alpha=0.05):
    #this test checks if there are several significant frequencies present.

    # na is set 8 as 4*d (where d is number of suspected significant frequencies) and d=2 is set as standard guess

    N= len(data)
    N2 = np.floor(N/2-0.5)
    N2 = N2.astype(int)

    rho2 = np.divide(np.power(np.fft.fft(data),2), N)
    rho2 = np.abs(rho2)
    rho2 = rho2[0:N2]

    print(rho2)
    
    ordR = np.sort(rho2, axis=None)
    sigmaEst = np.sum(ordR[1:N2-na])
    print(sigmaEst)

    #signPerInd = np.zeros(N2)
    signPerValVec = np.zeros(N2)


    for k in range(len(rho2)):
        ratioT = rho2[k] / sigmaEst
        prodQ = 1
        
        for k2 in range(N2-na):
            prodQ = prodQ / (1+k2*ratioT/(na+k2-1) )

        signPerValVec[k]=prodQ*N2
    
    
    # signPerInd = find( signPerValVec<alpha )-1;
    signPerInd = np.argwhere(signPerValVec < alpha)

    # signPerVal = signPerValVec( signPerInd+1 );
    signPerVal = signPerValVec[signPerInd]



    return signPerInd.size, signPerInd, signPerVal