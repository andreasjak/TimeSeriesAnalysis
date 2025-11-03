"""tsa_lth
=================

Package entry point for the Time Series Analysis library used in the
course. This file exposes the high-level modules and a selection of
convenience functions and classes for easier interactive use, while
keeping imports local to the package (relative imports).

The intent is to mirror the public API of the original tsa package
while avoiding absolute imports that may break when the package is
used as a local module.
"""

from . import analysis, modelling, tests, multivariate

from .analysis import (
    acf,
    pacf,
    xcorr,
    plotACFnPACF,
    plot_cum_per,
    kovarians,
    ccf,
    mat2pd,
    mat2np,
)

from .modelling import (
    PEM,
    filter,
    difference,
    estimateBJ,
    estimateARMA,
    simulateARMA,
    simulate_model,
    predict_pem,
    MultiInputPEM,
)

from .tests import (
    whiteness_test,
    monti_test,
    lbp_test,
    ml_test,
    check_if_normal,
)

from .multivariate import (
    corrM,
    pacfM,
    lsVAR,
    lbp_test_multivariate,
    lbp_test as lbp_test_mv,
    var_select_order,
    likelihood_ratio_test,
)

__version__ = "1.0.0"

__all__ = [
    # modules
    'analysis', 'modelling', 'tests', 'multivariate',
    # analysis
    'acf', 'pacf', 'xcorr', 'plotACFnPACF', 'plot_cum_per', 'kovarians', 'ccf', 'mat2pd', 'mat2np',
    # modelling
    'PEM', 'filter', 'difference', 'estimateBJ', 'estimateARMA', 'simulateARMA', 'simulate_model', 'predict_pem', 'MultiInputPEM',
    # tests
    'whiteness_test', 'monti_test', 'lbp_test', 'ml_test', 'check_if_normal',
    # multivariate
    'corrM', 'pacfM', 'lsVAR', 'lbp_test_multivariate', 'lbp_test_mv', 'var_select_order', 'likelihood_ratio_test',
]
