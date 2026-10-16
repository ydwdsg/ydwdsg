# v0.1.0 experimental release checklist

- [x] Extract frozen kernels into a separate tree; no production code changed.
- [x] Synthetic, public fixtures; no NEP/FC/HDF5/scientific input redistribution.
- [x] CPU regression against an independent formula and native Phono3py C.
- [x] Historical performance scope and numerical evidence disclosed.
- [x] Completed 72-atom full-WTE comparison accepted by owner; measured differences and original strict-gate record retained.
- [x] Independent all-q C reference accurately marked pending.
- [x] BSD-3-Clause and upstream notices.
- [x] Three reproducible figures with machine-readable sanitized sources.
- [x] Installed-wheel CPU/native-C/CUDA tests: 24 passed, zero skipped; GPU sharing explicitly authorized. Final wheel SHA and logs are supplied with the local release.
- [x] Project author and contact confirmed: yangdewei, 1147398767@qq.com; upstream notices retained.
- [ ] Confirm institutional/group permission before public distribution, if applicable.
- [ ] Complete independent original-C full-grid reference (outside this release task).
- [ ] Repository owner review, public repository creation and push.

This directory is prepared locally. A local Git repository, wheel, source
distribution and release ZIP do not mean the project is publicly hosted or
published on GitHub/PyPI. No remote, token, account or deployment is configured.
Before public release review attribution, synthetic tests, disclosures and artifact
scan again. Experimental code availability is not approval to replace production.

Rebuild locally with `python -m build`, then
`python scripts/prepare_release.py --out ../my-new-v0.1.0-release`.
The output path must not exist and must be outside the source repository.
The release contains the wheel, sdist, public source ZIP, SHA256SUMS and a
relative-path per-file manifest. No local Git metadata is archived.
