# Installed-wheel integration validation

The release was tested from a wheel installed into a new, isolated target,
not imported from the development source tree. Tests were taken from the matching
source distribution. No production package was installed, upgraded or patched.

Environment: Linux, Python 3.10.21, Phono3py 3.31.1, Phonopy 3.5.1,
NumPy 2.2.5, CuPy 14.2.0, pytest 8.4.2, NVIDIA RTX 5080.
The small fixture test was explicitly authorized to share a GPU with an existing
task, with more than 2 GiB free device memory checked before execution. Shared
execution is correctness evidence, not an uncontended performance measurement.

The suite requires **24 passed, zero failed, zero errors and zero skipped**,
including five explicitly opt-in CUDA test cases. It covers:

- Full and compact FC3 on a public two-atom synthetic, non-Gamma fixture.
- Every triplet at that fixture q point: complex GPU/NumPy Fourier tensors and
  complete squared interactions against an independently invoked upstream C path.
- Three primitive sizes against an independent NumPy einsum contraction oracle,
  including negative, zero and cutoff-boundary frequencies and selected bands.
- NumPy complex Fourier against upstream Python, native nonzero masks, unchanged
  input arrays, shape/index guards, and the packed-constant allocation guard.
- Standalone synthetic examples using both NumPy and CuPy.

Unchanged criteria: complex Fourier rtol 1e-11 / atol 1e-14; squared interaction
vs native C rtol 1e-8 / atol 1e-20; contraction oracle rtol 1e-12 / atol 1e-18.
No tolerance was relaxed after test execution.

The accompanying `VALIDATION.json` records the exact final wheel and sdist SHA256,
test counts, versions and time. `SHA256SUMS.txt` covers the deliverable artifacts.
The early dependency-incomplete QA attempts were retained privately; they were
not numerical PASS evidence. Only the final all-pass run authorizes this local
experimental delivery.

This does not supply a complete WTE solver, validate Phono3py 4.x, establish
material convergence, or repeat private full-M14 gamma/kappa/performance tests.
Historical measurements and their independent-C-reference limitations remain
separate. Nothing has been uploaded to a public repository or package index.
