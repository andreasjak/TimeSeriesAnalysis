#Marcus Lindell & Casper Schwerin
import numpy as np

#anomaly checks using IQR or STD
def iqr (df_in, inplace=False):

    if ~inplace:
        df = df_in.copy() #should not perform operations on original dataframe
    else:  
        df = df_in

    Q1 = df.quantile(0.25)
    Q2 = df.quantile(0.5)
    Q3 = df.quantile(0.75)
    IQR = Q3-Q1
    index_of_outliers = []

    for i in range(len(df)):
        if (df[i] <(Q1-1.5*IQR)):
            #print(f'Lower outlier found at index {i} (value = {df[i]}).')
            index_of_outliers.append(i)
        if (df[i] >(Q3+1.5*IQR)):
            #print(f'Upper outlier found at index {i} (value = {df[i]}).')
            index_of_outliers.append(i)

    #setting outliers to median value
    df.loc[df<(Q1-1.5*IQR)] = Q2 # eller Q1-1.5*IQR dvs trubba av till precis på gränsen
    df.loc[df>(Q3+1.5*IQR)] = Q2 # eller Q3+1.5*IQR dvs trubba av till precis på gränsen

    return df, index_of_outliers


def std (df_in, n=3, inplace=False):
    
    if ~inplace:
        df = df_in.copy() #should not perform operations on original dataframe

    else:
        df = df_in

        
    n_sd = df.std() * n
    index_of_outliers = []
    for i in range(len(df)):
        if (df[i]<(np.mean(df)-n_sd)):
            #print(f'Lower outlier found at index {i} (value = {df[i]}).')
            index_of_outliers.append(i)
            
        if (df[i]>(np.mean(df)+n_sd)):
            #print(f'Upper outlier found at index {i} (value = {df[i]}).')
            index_of_outliers.append(i)

    #setting outliers to mean value
    df.loc[df<(np.mean(df)-n_sd)] = np.mean(df)-n_sd # eller np.mean(df)-n_sd dvs trubba av till precis på gränsen
    df.loc[df>(np.mean(df)+n_sd)] = np.mean(df)+n_sd # eller np.mean(df)+n_sd dvs trubba av till precis på gränsen

    return df, index_of_outliers