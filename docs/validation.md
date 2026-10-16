# Validation scope

PASS means a stated test passed; it is not scientific convergence or production
approval. PENDING is missing evidence. Original strict-gate records are retained
separately from later project-owner acceptance decisions.

Public automated checks use deterministic synthetic, nonphysical FCs, not private
SnP3 assets. They test complex phases, full and compact mappings, r0 averaging,
positive masses, cutoffs, reordered bands, no-conjugation normal coordinates,
nonmutation, native-C interactions and nonfinite input rejection. The independent
einsum oracle is deliberately limited to <=30 branches. CUDA tests are opt-in.

Frozen private-fixture 24-q gamma and modal κP/κC evidence is distinct from the
current packaging revision's synthetic tests. `public_results.json` records
source-report hashes for traceability without distributing their private contents.
Anonymized q row numbers are not physical q coordinates. No new physics was run
to prepare the release. The complete independent C full-grid reference is still
pending. The completed 72-atom M14 WTE representation comparison is owner-accepted
(PASS_OWNER_ACCEPTED), with maximum reported relative difference 0.349325%.
The original strict rtol 1e-7 / atol 1e-12 result remains FAIL in the metadata.

The adapter does not recompute gamma, invoke WTE, select q weights, add linewidth
floors, or set 2D thickness. It returns interactions and native conversion only.
Any custom caller must independently preserve those downstream choices.

Known unsupported cases: Phono3py !=3.31.1; r0-average=False; FC3-q symmetrization;
nonuniform/incomplete atom mapping; sparse shortest vectors; NAC/degenerate gauge
equivalence across separate solvers; automatic primitive/supercell FC remapping.
The private 72-atom mapping code/assets are not redistributed in this release;
its completed comparison is documented as measured evidence.
Cross-platform last-bit differences are not required to vanish. A failure must
not be repaired by changing frequencies, cutoffs, integration parameters, masks,
physical inputs or reference arrays merely to force the gate.
