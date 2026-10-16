"""Array comparisons without hiding small/zero reference values."""
import numpy as np


def compare(actual, expected, *, rtol=1e-8, atol=1e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not a.size:
        raise ValueError("Comparison requires identical nonempty shapes")
    if rtol < 0 or atol <= 0:
        raise ValueError("rtol >= 0 and atol > 0 are required")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return dict(passed=False, finite=False, rtol=rtol, atol=atol)
    error = np.abs(a - b)
    tolerance = atol + rtol * np.abs(b)
    peak = float(np.abs(b).max())
    nonzero = np.abs(b) > atol
    return dict(passed=bool(np.all(error <= tolerance)), finite=True,
                max_abs=float(error.max()),
                max_tolerance_ratio=float((error / tolerance).max()),
                max_error_over_peak=float(error.max() / peak) if peak else None,
                max_relative_nonzero=float((error[nonzero] / np.abs(b[nonzero])).max()) if nonzero.any() else None,
                near_zero_count=int((~nonzero).sum()), rtol=rtol, atol=atol)
