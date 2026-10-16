# Figure contract and review

Audience: experimental software repository, not a journal submission.
Three original single-axis plots, each 183 mm wide; 300 dpi PNG, editable SVG
and embedded-font PDF. No TIFF is required for this software release.

| Figure | Claim | Encoding and caveat | Rendered review |
|---|---|---|---|
| 01 | Local contraction and Fourier acceleration | Log-position timings; distinct markers; Fourier ratio excludes setup | Labels clear, positive log data, no grid crossing labels |
| 02 | Timing scopes are not interchangeable | Zero-baseline bars; each row explicitly names worker/segmented/fresh scope | No overlap; all three observed times retained; no derived overall speedup |
| 03 | Completed 72-atom representation precision, accepted by owner | Percent relative deviations on a common linear axis; 73-atom regression summarized below | Labels clear; all three deviations retained |

These are historical observations and diagnostic maxima, not repeated
independent experiments. No uncertainty intervals are invented. Zero/negative
values are not discarded from scientific data: only the explicitly positive
summary timings/ratios are plotted. The 24-row error table remains available.

Final QA: source preflight has no FAIL; three PDF glyph-floor checks pass 5 pt
(smallest glyph 5.6 pt); all three PDF collision checks pass after revision;
all three PNGs were visually inspected. Multipanel alignment is not applicable.
Static-source warnings concerning TIFF/600 dpi are explicitly not applicable
to this PNG/SVG/PDF repository target. Automated QA is not scientific certification.

Style: restrained horizontal comparison structure; simple log points were chosen
over the alternative stacked-bar and rose-ring gallery patterns, because timings
with different scopes must not be stacked or encoded as parts of one whole.
No gallery images or code are redistributed.
