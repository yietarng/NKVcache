"""Turn a request's resolved sub-context matches into batch-prep offsets.

Pure, ``Req``-independent functions (kept out of ``schedule_batch.py`` to
avoid a circular import, since that module imports this package) so they're
unit-testable without pulling in the scheduler. The actual eligibility gate
(which requests are simple enough to take this path at all -- no
multimodal, no logprobs, no DLLM, ...) lives in
``managers/schedule_batch.py: prepare_for_extend`` next to the rest of the
extend-batch bookkeeping it has to stay consistent with.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.subcontext.subcontext_types import (
    RecomputePlan,
    SubContextMatch,
)

# (query_start, query_end, match) -- match is carried through so a caller can
# recover entry.slots / entry.orig_position for the range without a second
# lookup.
ReusedRange = Tuple[int, int, SubContextMatch]


def reused_ranges_within(
    plans: Sequence[RecomputePlan], *, extend_start: int, extend_end: int
) -> List[ReusedRange]:
    """Absolute ``[start, end)`` ranges safe to treat as pure KV reuse this
    round.

    A match whose span isn't *entirely* inside ``[extend_start, extend_end)``
    is dropped rather than partially materialized -- e.g. chunked prefill
    can split a match across a chunk boundary, and reconciling a partial
    reuse across chunks needs bookkeeping this function doesn't have. A
    dropped match's tokens are just computed normally.
    """
    ranges: List[ReusedRange] = []
    for plan in plans:
        match = plan.match
        if match.query_start < extend_start or match.query_end > extend_end:
            continue
        for start, end in plan.reused_ranges():
            ranges.append((start, end, match))
    ranges.sort(key=lambda r: r[0])
    return ranges


def surviving_offsets(
    reused_ranges: Sequence[ReusedRange], *, extend_start: int, extend_end: int
) -> List[int]:
    """Absolute positions in ``[extend_start, extend_end)`` that still need
    a real forward pass: the glue between matches, plus each match's
    selectively-recomputed offsets (already excluded from
    ``reused_ranges``)."""
    reused_positions = set()
    for start, end, _match in reused_ranges:
        reused_positions.update(range(start, end))
    return [
        pos for pos in range(extend_start, extend_end) if pos not in reused_positions
    ]


def source_slot_for_position(match: SubContextMatch, position: int) -> int:
    """The source entry's physical slot backing absolute ``position``
    within ``match``."""
    return match.entry.slots[position - match.query_start]


class RequestSubcontextPlan(NamedTuple):
    """One request's split of its extend range into tokens that still need
    a real forward pass and tokens whose KV gets materialized instead.

    ``*_absolute`` positions are 0-indexed from the start of the request's
    own token sequence (what the model's RoPE positions must read);
    ``*_local`` offsets are 0-indexed from ``prefix_len`` (what indexes the
    request's own slice of the batch's flat, already-allocated
    ``out_cache_loc``, since that slice covers exactly
    ``[prefix_len, prefix_len + extend_len)``).
    """

    surviving_absolute: Tuple[int, ...]
    surviving_local: Tuple[int, ...]
    reused_local: Tuple[int, ...]
    reused_source_slots: Tuple[int, ...]
    reused_delta_positions: Tuple[float, ...]


def plan_request_extend(
    recompute_plans: Sequence[RecomputePlan], *, prefix_len: int, extend_len: int
) -> RequestSubcontextPlan:
    extend_start, extend_end = prefix_len, prefix_len + extend_len
    ranges = reused_ranges_within(
        recompute_plans, extend_start=extend_start, extend_end=extend_end
    )
    survive = surviving_offsets(ranges, extend_start=extend_start, extend_end=extend_end)

    reused_local: List[int] = []
    reused_source_slots: List[int] = []
    reused_delta_positions: List[float] = []
    for start, end, match in ranges:
        delta = float(match.delta_position)
        for pos in range(start, end):
            reused_local.append(pos - prefix_len)
            reused_source_slots.append(source_slot_for_position(match, pos))
            reused_delta_positions.append(delta)

    return RequestSubcontextPlan(
        surviving_absolute=tuple(survive),
        surviving_local=tuple(pos - prefix_len for pos in survive),
        reused_local=tuple(reused_local),
        reused_source_slots=tuple(reused_source_slots),
        reused_delta_positions=tuple(reused_delta_positions),
    )


class BatchSubcontextPlan(NamedTuple):
    """Batch-flattened result of applying each request's
    ``RequestSubcontextPlan`` to the batch's already-allocated
    ``out_cache_loc`` -- everything ``ForwardBatch.init_new`` (positions,
    out_cache_loc, extend_lens) and the KV materializer
    (``kv_materialize.materialize_reused_kv``) need, with no further
    per-request bookkeeping.
    """

    out_cache_loc: torch.Tensor
    extend_lens: List[int]
    positions: torch.Tensor
    materialize_source_slots: torch.Tensor
    materialize_dest_slots: torch.Tensor
    materialize_delta_positions: torch.Tensor


def build_batch_subcontext_plan(
    *,
    request_plans: Sequence[Optional[RequestSubcontextPlan]],
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    out_cache_loc: torch.Tensor,
) -> BatchSubcontextPlan:
    """Split each request's slice of the flat, already-allocated
    ``out_cache_loc`` (covering its full ``[prefix_len, prefix_len +
    extend_len)``, one slice per request in order -- the invariant
    ``alloc_for_extend``'s ``write_cache_indices`` establishes) into the
    subset that still needs a forward pass and the subset a materializer
    call will fill instead, then re-concatenates the forward-only subset
    across the whole batch. A request with no plan (``None``: feature off,
    ineligible, or no matches this round) passes through unchanged.
    """
    device = out_cache_loc.device
    per_req_loc = torch.split(out_cache_loc, list(extend_lens))

    survive_parts: List[torch.Tensor] = []
    new_extend_lens: List[int] = []
    position_parts: List[torch.Tensor] = []
    src_slots: List[int] = []
    dest_slots: List[int] = []
    deltas: List[float] = []

    for plan, prefix_len, extend_len, loc in zip(
        request_plans, prefix_lens, extend_lens, per_req_loc
    ):
        if plan is None:
            survive_parts.append(loc)
            new_extend_lens.append(extend_len)
            position_parts.append(
                torch.arange(prefix_len, prefix_len + extend_len, device=device)
            )
            continue

        survive_idx = torch.tensor(
            plan.surviving_local, dtype=torch.int64, device=loc.device
        )
        survive_parts.append(loc.index_select(0, survive_idx))
        new_extend_lens.append(len(plan.surviving_absolute))
        position_parts.append(
            torch.tensor(plan.surviving_absolute, dtype=torch.int64, device=device)
        )

        if plan.reused_local:
            reuse_idx = torch.tensor(
                plan.reused_local, dtype=torch.int64, device=loc.device
            )
            dest_slots.extend(loc.index_select(0, reuse_idx).tolist())
            src_slots.extend(plan.reused_source_slots)
            deltas.extend(plan.reused_delta_positions)

    return BatchSubcontextPlan(
        out_cache_loc=torch.cat(survive_parts),
        extend_lens=new_extend_lens,
        positions=torch.cat(position_parts),
        materialize_source_slots=torch.tensor(src_slots, dtype=torch.int64, device=device),
        materialize_dest_slots=torch.tensor(dest_slots, dtype=torch.int64, device=device),
        materialize_delta_positions=torch.tensor(
            deltas, dtype=torch.float32, device=device
        ),
    )
