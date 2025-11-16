# -*- coding: utf-8 -*-
# Integração numérica com fallback de numpy.trapezoid para numpy.trapz.

import numpy as np


def trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)
