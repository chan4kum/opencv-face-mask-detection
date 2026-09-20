# ADR 0002: Upscale small images before detection; report counts as lower bounds

**Status:** accepted

**Context.** Half of the dataset's faces are under 24 px. With YuNet's defaults it found only 57% of validation faces. The classifier was never the bottleneck (96.9% on ground-truth boxes).

**Decision.** Add an optional `upscale_to` step to the detector (images whose longest side is below the target are enlarged, up to 4x, with bicubic interpolation) and choose the value on the **validation** images only: score 0.7 and upscale-to 1024 raised recall to 78% at 0.77 false detections per image. Flag faces under 24 px as `low_resolution`, and never present counts as a compliance rate.

**Consequences.**
- (+) About 20 more faces per 100 are found; the trade-off table is in `training/tune_detector.py`'s output and EVALUATION.md.
- (-) Upscaling costs latency on small images and slightly lowers recall for mid-size faces at very high targets (1600).
- (-) A quarter of faces are still missed; documented prominently.
