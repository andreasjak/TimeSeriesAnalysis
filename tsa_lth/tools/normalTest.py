#Marcus Lindell & Casper Schwerin
from scipy.stats import normaltest
from scipy.stats import jarque_bera

#normal test
def normalTest(data, alpha=0.05):
    k2, p = normaltest(data)

    if p < alpha: # null hypothesis: x comes from a normal distribution
        print(f"The D'Agostino-Pearson's test indicates that the set is NOT normal distributed.")
    else:
        print(f"The D'Agostino-Pearson's test indicates that the set is NORMAL distributed.")

    jb, p = jarque_bera(data)

    if p < alpha:
        print(f"The Jarque-Bera test indicates that the set is NOT normal distributed.")
    else:
        print(f"The Jarque-Bera test indicates that the set is NORMAL distributed.")
    
