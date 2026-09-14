# Vendored SGLang

This directory is a vendored, pinned snapshot of [sgl-project/sglang](https://github.com/sgl-project/sglang),
used as the base for NOC's sub-context KV cache work (position-independent
KV reuse via a content-addressed sub-context index, RoPE delta-repositioning,
and EPIC/CacheBlend-style selective recomputation).

- Upstream: https://github.com/sgl-project/sglang
- Pinned commit: `2f5cc8e33e9717f8284ed48ffc7f012e7b4a4197`
- Vendored: 2026-09-14
- License: Apache License 2.0 (see `LICENSE` in this directory)

A handful of MoE Triton autotuning config files under
`python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/` were dropped
during vendoring because their filenames contain literal spaces/brackets
that broke the copy step. They are unrelated to this work (exotic
hardware-specific MoE tuning tables) and can be re-synced from upstream if
ever needed.

NOC-specific changes are layered directly on top of this tree rather than
kept as an external patch set, so `git log -- sglang/` on this repo is the
diff against the pinned commit above.
