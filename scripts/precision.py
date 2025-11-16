# -*- coding: utf-8 -*-
# Contextos de precisão e utilitários de estabilidade numérica.

import warnings
from decimal import localcontext, ROUND_HALF_EVEN
import mpmath as mp


def decimal_context(prec: int = 100):
    ctx = localcontext()
    ctx.prec = prec
    ctx.rounding = ROUND_HALF_EVEN
    return ctx


def set_mpmath_dps(dps: int = 100):
    mp.mp.dps = dps


def enable_strict_deprecations():
    warnings.simplefilter("error", DeprecationWarning)
