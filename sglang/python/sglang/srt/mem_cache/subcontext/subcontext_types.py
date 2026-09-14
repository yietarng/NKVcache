"""Data types for the sub-context KV cache.

A sub-context is a semantically independent token span (a tool-output
fragment, a RAG document chunk, a system-prompt block, ...) whose KV cache
can be reused wherever that span recurs in a later prompt, independent of
its position. See ``subcontext_index.py`` for how spans are found and
``layers/rotary_embedding/reposition.py`` for how a reused span's K is
corrected for its new position.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Optional, Tuple

import msgspec
import torch


class SubContextTag(msgspec.Struct, frozen=True, kw_only=True):
    """An explicit, caller-supplied sub-context boundary on one request.

    Produced from a request's ``subcontext_tags`` (see ``io_struct.py``);
    consumed by the scanner as a high-confidence match that skips content
    hashing on the query side.
    """

    subcontext_id: str
    start: int  # inclusive token offset into the request's token ids
    end: int  # exclusive


class SubContextEntry(msgspec.Struct, frozen=True, kw_only=True):
    """A registered, previously-computed sub-context KV chunk."""

    content_hash: str  # sha256 hex of the (page-aligned) token span
    token_ids: Tuple[int, ...]  # the span's own tokens, for verification
    orig_position: int  # absolute position this chunk was first computed at
    slots: Tuple[int, ...]  # physical KV-pool slot indices, one per token
    subcontext_id: Optional[str] = None  # caller-supplied label, if tagged

    @property
    def length(self) -> int:
        return len(self.token_ids)


class SubContextMatch(msgspec.Struct, frozen=True, kw_only=True):
    """One resolved match of a registered entry inside a new query."""

    entry: SubContextEntry
    query_start: int  # inclusive offset into the *new* request's tokens
    query_end: int  # exclusive
    source: Literal["explicit", "scanned"]

    @property
    def delta_position(self) -> int:
        """Position shift to apply to the entry's cached K (new - old)."""
        return self.query_start - self.entry.orig_position

    @property
    def length(self) -> int:
        return self.query_end - self.query_start


class RecomputePlan(msgspec.Struct, frozen=True, kw_only=True):
    """Per-match decision on which of its tokens get selectively recomputed.

    ``recompute_offsets`` are offsets relative to ``match.query_start``
    (EPIC/CacheBlend-style correction for cross-attention deviation: the
    reused chunk's K/V were originally computed next to different
    neighboring content, so a small subset is recomputed with real
    attention over the new context instead of trusted as-is).
    """

    match: SubContextMatch
    recompute_offsets: Tuple[int, ...]

    def reused_ranges(self) -> Tuple[Tuple[int, int], ...]:
        """Contiguous [start, end) ranges (in query token space) that stay
        pure KV reuse, i.e. the match span minus recomputed offsets."""
        recompute = set(self.recompute_offsets)
        ranges = []
        start = None
        for off in range(self.match.length):
            pos = self.match.query_start + off
            if off in recompute:
                if start is not None:
                    ranges.append((start, pos))
                    start = None
                continue
            if start is None:
                start = pos
        if start is not None:
            ranges.append((start, self.match.query_end))
        return tuple(ranges)


class SubcontextMaterializePlan(NamedTuple):
    """Batch-flattened work list for ``kv_materialize.materialize_reused_kv``:
    for every layer, copy+reposition K and copy V from ``source_slots`` into
    ``dest_slots`` by ``delta_positions``. Set on ``ScheduleBatch`` by
    ``prepare_for_extend``, consumed by ``managers/tp_worker.py`` right
    before the model forward."""

    source_slots: torch.Tensor
    dest_slots: torch.Tensor
    delta_positions: torch.Tensor
