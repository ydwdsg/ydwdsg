# Attribution and redistribution scope

The project-specific implementation and synthetic fixtures use BSD-3-Clause
(LICENSE). The NumPy/CuPy kernels were extracted and generalized from frozen,
locally developed regression-tested prototypes, not from running production
files. Original FC3 transformation and normal-coordinate conventions follow:

- Phono3py v3.31.1, `c/real_to_reciprocal.c`, copyright (C) 2015 Atsushi Togo.
- Phono3py v3.31.1, `c/reciprocal_to_normal.c`, copyright (C) 2015 Atsushi Togo.
- Upstream project copyright (c) 2015–2024, Phonopy.

The applicable BSD-3-Clause notices are retained in
`licenses/PHONO3PY-BSD-3-Clause.txt`. This includes the numerical translations
in `fourier.py`, `contraction.py`, `reference.py` and the upstream interface in
`phono3py_adapter.py`; new adaptations remain under BSD-3-Clause.

Primary sources:

- [Tagged upstream license](https://github.com/phonopy/phono3py/blob/v3.31.1/LICENSE)
- [Real-to-reciprocal source](https://github.com/phonopy/phono3py/blob/v3.31.1/c/real_to_reciprocal.c)
- [Reciprocal-to-normal source](https://github.com/phonopy/phono3py/blob/v3.31.1/c/reciprocal_to_normal.c)

NumPy, CuPy, Phonopy, Phono3py, Matplotlib and pytest are separately installed
dependencies; their sources/binaries are not vendored. Their own licenses apply.
No custom WTE plugin is bundled or relicensed. The user must supply an appropriately
licensed downstream WTE integration if needed.

Benchmark summaries are released only as software performance and numerical-error
metadata. No private structure, potential, force constant, absolute material
conductivity, linewidth array, or unpublished scientific input is redistributed.
