#Marcus Lindell & Casper Schwerin
import numpy as np

def get_metrics(error):
    
    mse = ((error)**2).mean()

    mae = (np.abs(error)).mean()

    var = np.var(error)

    return mse, mae, var