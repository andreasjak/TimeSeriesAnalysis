#Marcus Lindell & Casper Schwerin
from statsmodels.tsa.stattools import adfuller #augmented dickyfullertest
import pandas as pd

#augmented dicky-fuller test
def adf_test(data):
    adft = adfuller(data, autolag="AIC")
    output_df = pd.DataFrame({  "Values": 
                                    [adft[0],adft[1],adft[2],adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']], 
                                "Metric": 
                                        ["Test Statistics", "p-value", "No. of lags used", "Number of observations used", 
                                        "critical value (1%)", "critical value (5%)", "critical value (10%)"]
                            })
    print(output_df)
    #null hypothesis is NON-stationary. IF p>0.05 ==> NON-stationary (hypothesis cannot be rejected)