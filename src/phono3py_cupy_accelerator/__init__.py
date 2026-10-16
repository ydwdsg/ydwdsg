"""Experimental acceleration kernels. No automatic backend registration."""

from .contraction import contract
from .fourier import FourierBatch
from .validation import compare

__version__ = "0.1.0"
__all__ = ["contract", "FourierBatch", "compare"]
