"""Read-only adapter for the tested Phono3py 3.31.1 Interaction layout.

No monkey patch, installation change, or WTE plugin is performed. Keep the
Interaction and its FC arrays unchanged for the lifetime of FourierBatch.
"""
from importlib.metadata import version
import numpy as np


def validate_interaction(pp):
    if version("phono3py") != "3.31.1":
        raise RuntimeError("v0.1.0 supports Phono3py 3.31.1 only; other APIs are unvalidated")
    required = ("_p2s", "_s2p", "_fc3", "_fc3_nonzero_indices", "_all_shortest",
                "_multi", "_svecs", "_bz_grid", "_make_r0_average", "_masses")
    missing = [name for name in required if not hasattr(pp, name)]
    if missing:
        raise ValueError(f"Interaction layout is incompatible: {missing}")
    n, ns = len(pp._p2s), len(pp._s2p)
    if n == 0 or len(pp._masses) != n or ns % n:
        raise ValueError("Invalid primitive/supercell sizes")
    if pp._fc3.shape not in ((n, ns, ns, 3, 3, 3), (ns, ns, ns, 3, 3, 3)):
        raise ValueError("Full or compact FC3 shape required")
    if pp._fc3.dtype != np.float64 or not np.isfinite(pp._fc3).all():
        raise ValueError("FC3 must be finite float64")
    if pp._multi.shape != (ns, n, 2) or np.any(pp._multi[:, :, 0] <= 0):
        raise ValueError("Dense shortest-vector multiplicity layout required")
    if pp._all_shortest is None or pp._all_shortest.shape != (n, ns, ns):
        raise ValueError("Native all_shortest flags must be initialized")
    if pp._fc3_nonzero_indices.shape != pp._fc3.shape[:3]:
        raise ValueError("Native FC3 nonzero mask shape mismatch")
    if getattr(pp, "_symmetrize_fc3q", False):
        raise ValueError("symmetrize_fc3q=True is not supported in v0.1.0")
    groups = [np.flatnonzero(pp._s2p == s) for s in pp._p2s]
    if any(len(g) != ns // n for g in groups) or len(np.unique(np.concatenate(groups))) != ns:
        raise ValueError("Mapping must cover every supercell atom exactly once")


def evaluate_triplet(pp, triplet, *, backend="numpy", fourier=None):
    """Return full |V3|² with native cutoff and conversion, plus timings.

    ``pp`` must already have phonons solved at these three BZ indices. Nothing
    is registered with Phono3py; feed results into your own validated integration
    driver. Do not replace a production driver without independent regression.
    """
    from .fourier import FourierBatch
    from .contraction import contract
    validate_interaction(pp)
    triplet=validate_triplet(pp,triplet)
    if fourier is None:
        fourier = FourierBatch(pp, backend)
    if fourier.pp is not pp or fourier.backend != backend:
        raise ValueError("Fourier cache belongs to another Interaction/backend")
    cube = fourier(triplet)
    freq, eig, done = pp.get_phonons()
    ids = np.asarray(triplet, dtype=int)
    if not np.all(done[ids]):
        raise ValueError("Solve existing phonons first; adapter does not run solvers")
    values, timing = contract(cube, np.ascontiguousarray(eig[ids]),
                              np.ascontiguousarray(freq[ids]), pp._masses,
                              pp.band_indices, pp.cutoff_frequency,
                              pp.unit_conversion_factor, backend=backend)
    return values, dict(contraction=timing, fourier=fourier.metrics())


def native_triplet(pp, triplet):
    """Independent original-C interaction oracle for ONE small triplet.

    Uses the upstream Python extension rather than unguarded ctypes symbols.
    No pp interaction_strength array, gamma, or FC is replaced.
    """
    validate_interaction(pp)
    triplet=validate_triplet(pp,triplet)
    import phono3py._phono3py as native
    ids = np.ascontiguousarray([triplet], dtype=np.int64)
    d = 3 * len(pp._masses)
    output = np.zeros((1, len(pp.band_indices), d, d), dtype=np.float64)
    zeros = np.zeros(output.shape, dtype=np.byte)
    freq, eig, done = pp.get_phonons()
    if not np.all(done[ids]):
        raise ValueError("Phonons must be initialized")
    native.interaction(output, zeros, freq, eig, ids,
                       pp._bz_grid.addresses, pp._bz_grid.D_diag, pp._bz_grid.Q,
                       pp._fc3, pp._fc3_nonzero_indices, pp._svecs, pp._multi,
                       pp._masses, pp._p2s, pp._s2p, pp.band_indices,
                       0, int(pp._make_r0_average), pp._all_shortest,
                       pp.cutoff_frequency, 0)
    return output[0] * pp.unit_conversion_factor


def validate_triplet(pp,triplet):
    ids=np.asarray(triplet)
    if ids.shape!=(3,) or not np.issubdtype(ids.dtype,np.integer):
        raise ValueError('Three integer BZ-grid indices required')
    if np.any(ids<0) or np.any(ids>=len(pp._bz_grid.addresses)):
        raise ValueError('Triplet index outside BZ grid')
    return np.ascontiguousarray(ids,dtype=np.int64)
