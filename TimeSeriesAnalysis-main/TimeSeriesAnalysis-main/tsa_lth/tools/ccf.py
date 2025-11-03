#Marcus Lindell & Casper Schwerin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as smt

def ccf_plot(data_1, data_2):
    df = pd.DataFrame([data_1,data_2]).T
    df.columns = ['A', 'B'] 

    #backwards = smt.ccf(df['B'], df['A'], unbiased=False)[::-1] #unbiased är deprecated och ger future warnings, men sparat syntax här
    #forwards = smt.ccf(df['A'], df['B'], unbiased=False)
    backwards = smt.ccf(df['B'], df['A'], adjusted=False)[::-1]
    forwards = smt.ccf(df['A'], df['B'], adjusted=False)

    #ccf_output = np.r_[backwards[:-1], forwards]
    #plt.stem(range(-len(ccf_output)//2+1, len(ccf_output)//2+1), ccf_output)
    ccf_output = np.r_[backwards[len(backwards)-31:-1], forwards[:30]]
    plt.stem(range(-30,30),ccf_output)
    plt.xlabel('Lag')
    plt.ylabel('CCF')

        # 95% UCL / LCL
    plt.axhline(-1.96/np.sqrt(len(df)), color='k', ls='--') 
    plt.axhline(1.96/np.sqrt(len(df)), color='k', ls='--')
    plt.title("Crosscorrelation between the OMX dataset and sick absence")
    plt.tight_layout()
    plt.savefig("/Users/casperschwerin/Documents/SKOLA/År 5/exarbete/Figures/ccf_plot_omx.png")

    maxValue = max(ccf_output.min(), ccf_output.max(), key=abs)
    lag = np.where(ccf_output==maxValue)[0][0]

    #In order to display correct number for the delay
    if (lag  < (len(ccf_output)//2 +1)):
        lag = lag - (len(ccf_output)//2)
    else:
        lag = lag - len(ccf_output)//2

    #Explaining what rthe esults mean
    if (lag < 0):
        print(f'The delay seems to be {lag}, which suggests that the second data input is AHEAD by {-lag} lags.')
    else:
        print(f'The delay seems to be {lag}, which suggests that the second data input is BEHIND by {lag} lags.')

    #Further explaining of correlation
    if (maxValue < 0):
        print(f'The correlation at this lag is {maxValue}, which suggests that they are NEGATIVELY correlated.')
    else:
        print(f'The correlation at this lag is {maxValue}, which suggests that they are POSITIVELY correlated.')
    
    return lag, ccf_output