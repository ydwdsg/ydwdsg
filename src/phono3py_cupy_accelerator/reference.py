"""Independent, deliberately small NumPy einsum oracle; not a benchmark path."""
import numpy as np


def contract_reference(fc3, eigenvectors, frequencies, masses, bands, cutoff, unit_conversion):
    d = 3 * len(masses)
    if d > 30:
        raise ValueError("The direct einsum oracle is restricted to <=30 branches")
    e = np.asarray(eigenvectors) / np.sqrt(np.repeat(masses, 3))[None, :, None]
    b = np.asarray(bands, dtype=int)
    z = np.einsum("ijk,ia,jb,kc->abc", fc3, e[0][:, b], e[1], e[2], optimize=False)
    f0, f1, f2 = np.asarray(frequencies)
    valid = (f0[b, None, None] > cutoff) & (f1[None, :, None] > cutoff) & (f2[None, None, :] > cutoff)
    den = f0[b, None, None] * f1[None, :, None] * f2[None, None, :]
    return np.where(valid, np.abs(z)**2 / np.where(valid, den, 1) * unit_conversion, 0)
