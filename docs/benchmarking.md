# Honest benchmarking

Run only on explicitly available resources. Set thread counts and affinity in
your shell before Python starts. This project neither schedules jobs nor changes
other processes' affinity. A small default fixture is intentional; 219 branches
requires `--atoms 73 --allow-large` and enough host/device memory. Do not infer
the historical private performance from the synthetic timings.

Record device model, driver/runtime, package versions, physical cores, SMT,
thread settings, CPU binding, BLAS backend, contention and input dimensions.
Warm up and synchronize the CUDA stream; keep first-call time separate. Report
transfers, preparation, steady-state wall, total process wall, CPU time, host RSS
and sampled device process memory separately. A CUDA event stream span is not
an exclusive-active-kernel profile. Pool reservation is not actual process peak.

For full pipeline comparisons use identical inputs and numerical protocol,
independent fresh processes, matched resource limits and no old gamma caches.
Sum-of-q timings omit process launch, preparation, pauses, checkpoint and WTE
overhead; label them segmented, never fresh cold-start wall.

The included historical points are observations, not enough independent runs to
justify statistical uncertainty bars. No invented error bars are drawn. Stage
timing uses log-position points, not truncated logarithmic bars. Pipeline bars
have a zero baseline and explicitly different scopes. The accuracy figure plots
modal diagnostic summaries and the owner-accepted 72-atom representation
differences, including the unchanged earlier strict-gate record.
