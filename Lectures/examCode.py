# Time series analysis - Examination code.
# Lund University, Lund, Sweden
#
# This script shows the format used to evaluate a predictor at the examination, using the
# Helsinki energy data and the model from example code 28 (code28.ipynb).
#
# The prediction function must have the format pred_code_A_grpXXX(data, k). It does not take the
# model parameters as input, so these need to be stored inside the function. The function
# receives the data in the original domain, so any needed transform has to be done inside the
# function. You will not be told which data is used to test the predictor.
#
# Note: This is a Python version based on code28 and examinePrediction.m.
#
# Reference:
#   "An Introduction to Time Series Modeling", 4th ed, by Andreas Jakobsson
#   Studentlitteratur, 2021
#
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TimeSeriesAnalysis-main', 'TimeSeriesAnalysis-main')))
from tsa_lth.analysis import examine_prediction
from tsa_lth.modelling import polydiv


def pred_code_A_grp001(data, k):
    """
    Forms the k-step prediction of the data using the model from example code 28.

    Parameters:
    - data (array-like): The data, in the original domain.
    - k (int): Prediction horizon.

    Returns:
    - yhatk (ndarray): The k-step prediction, in the original domain. yhatk[t] is the
      prediction of data[t] formed using data up to time t-k.
    """
    sday = 24                                           # Daily season.
    sweek = 168                                         # Weekly season.

    # The model parameters, as estimated in code28 on the transformed training data.
    A = np.array([1, -0.4923, -0.2998])
    C = np.zeros(193)
    C[[0, 24, 168, 192]] = [1, -0.6152, -0.6879, 0.4495]

    # Add the seasonal differentiations to the model.
    dayPoly = np.concatenate([[1], np.zeros(sday-1), [-1]])
    weekPoly = np.concatenate([[1], np.zeros(sweek-1), [-1]])
    Am = np.convolve(np.convolve(A, dayPoly), weekPoly)
    Cm = C

    # Form the prediction in the transformed domain, then transform back.
    F, G = polydiv(Cm, Am, k)
    yhatk = signal.lfilter(G, Cm, np.sqrt(data))
    return yhatk**2


if __name__ == '__main__':
    k = 1                                               # Prediction horizon.
    sweek = 168

    powerload = io.loadmat(os.path.join(os.path.dirname(__file__), '..', 'data', 'dataHelsinki.mat'))['powerload'].flatten()

    # This is the test data. At the examination, you will not be told which data is used.
    testDataInd = np.arange(7155, 7155 + 4*sweek)

    yhatk = pred_code_A_grp001(powerload, k)

    plt.figure(figsize=(10, 5))
    plt.plot(powerload[testDataInd], label='Powerload')
    plt.plot(yhatk[testDataInd], label='Prediction')
    plt.legend()
    plt.title(f'{k}-step prediction in the original domain')
    plt.xlabel('Time')
    plt.show()

    examine_prediction(powerload[testDataInd], yhatk[testDataInd], k)
