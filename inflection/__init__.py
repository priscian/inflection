"""
inflection: Find inflection points of curves

A Python implementation of methods for finding inflection points,
based on the R inflection package by Demetris T. Christopoulos.
"""

__version__ = "0.2.0"
__author__ = "Python port of Demetris T. Christopoulos's R package"

from .core import (
    check_curve,
    ede,
    ese,
    bede,
    bese,
    edeci,
    findiplist,
    uik,
    lin2,
    findipl,
)

__all__ = [
    "check_curve",
    "ede",
    "ese",
    "bede",
    "bese",
    "edeci",
    "findiplist",
    "uik",
    "lin2",
    "findipl",
]
