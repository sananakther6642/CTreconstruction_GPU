# Experimentation

Investigations that produced real, measured, report-worthy findings but
were not merged into `features`/`main` — either negative results, or
alternatives superseded by what shipped. Kept here (rather than only on
their unmerged branches) so the code/writeup is visible alongside the
main submission for grading and the report.

- **`hybrid-precision/`** — fp16-speed sampler + fp32 correction for
  `gpu-img`/`gpu-opt`; found redundant once `gpu-buf` was recognized as
  already meeting the precision requirement.
- **`fdk/`** — single-pass analytical FDK reconstruction as an alternative
  to iterative MLEM; ~240x faster but ~3500x worse MSE — a genuine
  speed/quality trade-off, not a replacement.
- **`perf-v2/`** — FDK-as-initializer, Huber-prior regularization,
  log-domain momentum, and an SQS evaluation. OSEM (from the same
  branch) is the one piece that did ship — see the main README. The
  rest are documented negative results / non-implementations.

Each subfolder has its own README with the specific findings and, where
the branch shares git history with `features`, copies of the actual
changed source files. `perf-v2` has no shared history with `features`
(diverged early, never rebased) so only the measured-findings writeup is
preserved there — check out the `perf-v2` branch directly for its code.
