
"""
NetTools
========
Module which groups utilities to work with networks.

"""

import version
import release

from tests import test_netsym, test_netstructure
## Not inform about warnings
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
warnings.simplefilter("ignore")


def test():
    test_netsym.test()
    test_netstructure.test()
