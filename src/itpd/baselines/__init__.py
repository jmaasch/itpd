"""Baselines that use the known time order: IAMB, IPaDL, FullCI and Lasso.

Like ITPD, every baseline takes the lag-1 self edge of a target as known and
searches the other parents among all earlier nodes.
"""
from ._targets import Result
from .full_conditioning import full_conditioning
from .iamb import iamb
from .ipadl import ipadl
from .lasso import lasso

__all__ = ["iamb", "ipadl", "full_conditioning", "lasso", "Result"]
