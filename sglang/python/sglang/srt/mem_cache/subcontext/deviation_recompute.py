"""Boundary Recompute Zone (BRZ): EPIC/CacheBlend-style selective
recomputation for reused sub-contexts.

Repositioning a cached chunk's K (``layers/rotary_embedding/reposition.py``)
fixes its *position* but not its *content*: the chunk's K/V were originally
computed while attending to whatever preceded it the first time it was
seen, and now sit next to different neighbors. Pure reuse is therefore an
approximation, worst right at a stitched boundary. Following CacheBlend/EPIC,
a small window of tokens at each boundary gets forced through a real
forward pass -- attending to the actual new context -- to bound that
error, while the rest of each chunk stays pure KV reuse.

For adjacent segments A and B, the repair region is defined as::

    BRZ(A, B) = Last_k(A) union First_k(B),    k in {16, 32, 64}

This module only decides *which offsets* get recomputed; the scheduler
integration (``managers/schedule_batch.py``) is what turns that decision
into an actual forward pass.
"""

from __future__ import annotations

from typing import List, Sequence

from sglang.srt.mem_cache.subcontext.subcontext_types import RecomputePlan, SubContextMatch

VALID_BRZ_WINDOWS = (0, 16, 32, 64)


def plan_boundary_recompute_zones(
    matches: Sequence[SubContextMatch], *, k: int
) -> List[RecomputePlan]:
    """Apply BRZ to every match in one request, in query order.

    ``First_k(B)``: every match's own leading ``k`` tokens are always
    recomputed -- whatever precedes a reused chunk in the new sequence
    (the literal prefix, freshly computed glue, or another reused chunk)
    is essentially never what preceded it the first time it was computed.

    ``Last_k(A)``: a match's own trailing ``k`` tokens are additionally
    recomputed only when the *next* match starts exactly where this one
    ends (``matches[i].query_end == matches[i + 1].query_start``) -- a
    genuine stitched boundary between two reused chunks with no freshly
    computed glue between them. A match followed by glue, or by nothing
    (end of the extend range), needs no trailing repair: glue is already
    computed fresh against this match's true tail, so there is no second
    boundary to repair.

    ``matches`` must be sorted by ``query_start`` (as the scanner already
    produces them) and non-overlapping (the scanner's own invariant).
    ``k`` only needs to be non-negative here -- the spec's ``{0, 16, 32,
    64}`` (``VALID_BRZ_WINDOWS``) is a config-surface choice
    (``--subcontext-brz-window``, validated in
    ``Scheduler.maybe_init_subcontext_index``), not a constraint the
    algorithm itself requires.
    """
    assert k >= 0, k
    plans = []
    for i, match in enumerate(matches):
        offsets = set(range(min(k, match.length)))
        stitched_to_next = (
            i + 1 < len(matches) and matches[i + 1].query_start == match.query_end
        )
        if stitched_to_next:
            trailing_start = max(0, match.length - k)
            offsets.update(range(trailing_start, match.length))
        plans.append(RecomputePlan(match=match, recompute_offsets=tuple(sorted(offsets))))
    return plans
