"""EPIC/CacheBlend-style selective recomputation for reused sub-contexts.

Repositioning a cached chunk's K (``layers/rotary_embedding/reposition.py``)
fixes its *position* but not its *content*: the chunk's K/V were originally
computed while attending to whatever preceded it the first time it was
seen, and now sit next to different neighbors. Pure reuse is therefore an
approximation. Following CacheBlend/EPIC, a small subset of each chunk's
tokens gets forced through a real forward pass -- attending to the actual
new context -- to bound that error, while the rest stays pure KV reuse.

This module only decides *which offsets* get recomputed; the scheduler
integration (``managers/schedule_batch.py``) is what turns that decision
into an actual forward pass.
"""

from __future__ import annotations

import math

from sglang.srt.mem_cache.subcontext.subcontext_types import RecomputePlan, SubContextMatch


def plan_prefix_fraction(
    match: SubContextMatch, *, recompute_ratio: float, min_recompute: int = 1
) -> RecomputePlan:
    """Recompute the first ``ceil(length * recompute_ratio)`` tokens of the
    match. Cheap and deterministic: a chunk's leading tokens are the ones
    whose attention pattern most reflects what used to precede them, so
    they carry most of the cross-attention deviation from being relocated.
    """
    assert 0.0 <= recompute_ratio <= 1.0, recompute_ratio
    if recompute_ratio == 0.0:
        return RecomputePlan(match=match, recompute_offsets=())
    count = max(min_recompute, math.ceil(match.length * recompute_ratio))
    count = min(count, match.length)
    return RecomputePlan(match=match, recompute_offsets=tuple(range(count)))


def plan_none(match: SubContextMatch) -> RecomputePlan:
    """Pure reposition-and-reuse, no correction (PromptCache-style)."""
    return RecomputePlan(match=match, recompute_offsets=())
