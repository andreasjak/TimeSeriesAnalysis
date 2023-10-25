#Marcus Lindell & Casper Schwerin
from tools.ml import ml_test
from tools.lbp import lbp_test
from tools.monti import monti_test
from tools.sign import sign_test

#checks the whiteness of data using 4 tests
def whitenessTest(x, k = 24, alpha=0.05):
    print(f'Whiteness test with {alpha*100}% significance')

    S, Q, chiV = lbp_test(x, k=k)
    print(f'\tLjung-Box-Pierce test: {S} (white if {Q} < {chiV})')

    S, Q, chiV = ml_test(x, k=k)
    print(f'\tMcLeod-Li test:        {S} (white if {Q} < {chiV})')

    S, Q, chiV = monti_test(x, k=k)
    print(f'\tMonti test:            {S} (white if {Q:} < {chiV})')

    nRatio, confInt = sign_test(x)
    print(f'\tSign test:             {1 if nRatio > confInt[0] and nRatio < confInt[1] else 0} (white if {nRatio} in [{confInt[0]}, {confInt[1]}])')