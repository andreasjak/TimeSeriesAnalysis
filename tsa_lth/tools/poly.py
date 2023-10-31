#Marcus Lindell & Casper Schwerin
import numpy as np
from scipy.signal import deconvolve

#polynomial division
def polydiv(C,A,k=1):

    #equivalent(is it?) of MATLAB:S "equalLength" command
    if(len(A)> len(C)):
        diff = len(A) - len(C)
        C = np.concatenate((C, np.zeros(diff)))
    elif(len(A)< len(C)):
        diff = len(C) - len(A)
        A = np.concatenate((A, np.zeros(diff)))

    # MATLAB => [F, G] = polydiv( dataModel.C, dataModel.A, k );
    Q,R = deconvolve(np.convolve(np.concatenate((np.ones(1), np.zeros((k-1)))), C), A) 

    return Q,R

