"""CPU-side scan: find registered sub-contexts inside a new token sequence.

Runs once per request, over the *unmatched suffix* left after the normal
prefix-radix match (see ``managers/schedule_batch.py``). Two sources of
matches, combined:

1. Explicit tags the caller attached to the request (system prompt, a named
   tool-output span, a RAG fragment id, ...) -- checked directly against the
   index, no scanning needed.
2. Automatic detection: a greedy, longest-match-first scan using the index's
   probe-hash buckets as a candidate filter, so most positions cost one
   hash + one dict lookup rather than a token-by-token compare.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from sglang.srt.mem_cache.subcontext.subcontext_index import (
    PROBE_LEN,
    SubContextIndex,
    hash_token_span,
    probe_hash,
)
from sglang.srt.mem_cache.subcontext.subcontext_types import (
    SubContextEntry,
    SubContextMatch,
    SubContextTag,
)


def _verify_full_match(
    entry: SubContextEntry, token_ids: Sequence[int], start: int
) -> bool:
    end = start + entry.length
    if end > len(token_ids):
        return False
    return tuple(token_ids[start:end]) == entry.token_ids


def _match_explicit_tags(
    index: SubContextIndex,
    token_ids: Sequence[int],
    tags: Sequence[SubContextTag],
) -> List[SubContextMatch]:
    matches = []
    for tag in sorted(tags, key=lambda t: t.start):
        if tag.start < 0 or tag.end > len(token_ids) or tag.start >= tag.end:
            continue
        span = tuple(token_ids[tag.start : tag.end])
        content_hash = hash_token_span(span)
        entry = index.lookup(content_hash)
        if entry is None or entry.length != (tag.end - tag.start):
            continue
        matches.append(
            SubContextMatch(
                entry=entry, query_start=tag.start, query_end=tag.end, source="explicit"
            )
        )
    return matches


def _longest_candidate_match(
    index: SubContextIndex, token_ids: Sequence[int], pos: int, max_end: int
) -> SubContextMatch | None:
    probe = probe_hash(token_ids, pos)
    if probe is None:
        return None
    best: SubContextEntry | None = None
    for candidate in index.candidates_for_probe(probe):
        if candidate.length <= (best.length if best else -1):
            continue
        if pos + candidate.length > max_end:
            continue
        if _verify_full_match(candidate, token_ids, pos):
            best = candidate
    if best is None:
        return None
    return SubContextMatch(
        entry=best, query_start=pos, query_end=pos + best.length, source="scanned"
    )


def scan(
    index: SubContextIndex,
    token_ids: Sequence[int],
    *,
    explicit_tags: Sequence[SubContextTag] = (),
    start: int = 0,
    end: Optional[int] = None,
) -> List[SubContextMatch]:
    """Non-overlapping matches over ``token_ids[start:end]``, sorted by
    query_start.

    Explicit tags take priority over their span; automatic scanning fills
    in the rest, greedily preferring the longest verified match at each
    position so a big chunk isn't shadowed by a short one that happens to
    share a prefix. ``start`` excludes the region a caller already resolved
    some other way (e.g. the ordinary prefix-radix match, which is strictly
    cheaper reuse than a repositioned sub-context and always wins there).
    ``end`` (default ``len(token_ids)``) caps how far a match may reach --
    the caller must leave at least the request's own last token out of
    ``end`` (mirroring ``Req._compute_max_prefix_len``'s ``input_len - 1``
    rule for the ordinary prefix match): a match is never allowed to cover
    every token, or the request would have no token left to compute a
    logit and sample from.
    """
    n = len(token_ids)
    if end is None:
        end = n
    covered = bytearray(n)
    matches: List[SubContextMatch] = []

    for match in _match_explicit_tags(index, token_ids, explicit_tags):
        if match.query_start < start or match.query_end > end:
            continue
        for i in range(match.query_start, match.query_end):
            covered[i] = 1
        matches.append(match)

    pos = start
    while pos <= end - PROBE_LEN:
        if covered[pos]:
            pos += 1
            continue
        found = _longest_candidate_match(index, token_ids, pos, end)
        if found is None:
            pos += 1
            continue
        matches.append(found)
        for i in range(found.query_start, found.query_end):
            covered[i] = 1
        pos = found.query_end

    matches.sort(key=lambda m: m.query_start)
    return matches
