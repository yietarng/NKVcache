"""Content-addressed index of registered sub-context KV chunks.

Unlike ``RadixCache`` (keyed by root-to-node prefix path), this index is
keyed purely by content hash: a sub-context can be looked up regardless of
where it sits in a request's token sequence. See ``subcontext_scanner.py``
for how a new request's tokens are matched against it.

Physical storage is a fixed-size ring buffer of KV-pool slots, reserved
once at startup (``Scheduler.maybe_init_subcontext_index``) and never
returned to the main allocator -- registered chunks are *copies* living in
their own space, not aliases into the ordinary radix-cache pool. Aliasing
would need this index to be notified whenever the radix cache's own
eviction policy reclaims an overlapping node, which it has no hook for
today; a dedicated ring buffer sidesteps that coordination problem
entirely at the cost of some fixed memory (``--subcontext-kv-cache-tokens``).
"""

from __future__ import annotations

import hashlib
import struct
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.mem_cache.subcontext.subcontext_types import SubContextEntry

# Fixed-length probe used as a cheap first-level filter during scanning: the
# hash of a span's first PROBE_LEN tokens narrows candidates before a full
# token-by-token verification. Spans shorter than this can only be found via
# explicit tagging, not automatic scanning.
PROBE_LEN = 8


def hash_token_span(token_ids: Sequence[int]) -> str:
    packed = struct.pack(f"<{len(token_ids)}q", *token_ids)
    return hashlib.sha256(packed).hexdigest()


def probe_hash(token_ids: Sequence[int], offset: int) -> Optional[str]:
    """Hash of ``token_ids[offset : offset + PROBE_LEN]``, or None if the
    span from ``offset`` is shorter than the probe window."""
    end = offset + PROBE_LEN
    if end > len(token_ids):
        return None
    return hash_token_span(token_ids[offset:end])


class SubContextIndex:
    """Registers and looks up sub-context KV chunks by content hash.

    Thread-safe for the scheduler's single-writer-many-reader access
    pattern (register from the request-finished path, lookup from the
    request-scheduling path).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, SubContextEntry]" = OrderedDict()
        self._by_probe: Dict[str, List[str]] = {}
        self._refcount: Dict[str, int] = {}
        self._hash_to_probe: Dict[str, str] = {}

        self._slots: List[int] = []
        self._cursor = 0
        # content_hash occupying ring position i, or None if free.
        self._ring_owner: List[Optional[str]] = []
        # content_hash -> (ring_start, length), the range register() reserved.
        self._ring_span: Dict[str, Tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def bind_slots(self, slots: Sequence[int]) -> None:
        """One-time setup: give the index the physical KV-pool slots it
        owns (reserved once via the token allocator and never freed back).
        Must be called before any ``register()``."""
        with self._lock:
            assert not self._slots, "bind_slots must be called exactly once"
            self._slots = list(slots)
            self._ring_owner = [None] * len(self._slots)

    @property
    def capacity_tokens(self) -> int:
        return len(self._slots)

    def register(
        self,
        *,
        token_ids: Sequence[int],
        orig_position: int,
        subcontext_id: Optional[str] = None,
    ) -> Optional[List[int]]:
        """Reserve ring-buffer slots for ``token_ids`` and insert its entry.

        Returns the destination slot indices for the caller to copy K/V
        into, or ``None`` when there is nothing to copy: the content is
        already registered (a no-op touch), it's longer than the whole
        pool, or no unreferenced room could be found (the pool is full of
        in-flight entries -- registration is simply skipped, never forced
        by evicting something a live request is still consuming).
        """
        token_ids = tuple(token_ids)
        content_hash = hash_token_span(token_ids)
        with self._lock:
            if content_hash in self._entries:
                self._entries.move_to_end(content_hash)
                return None
            length = len(token_ids)
            if length == 0 or length > len(self._slots):
                return None
            ring_positions = self._reserve_ring_range_locked(length)
            if ring_positions is None:
                return None

            dest_slots = [self._slots[p] for p in ring_positions]
            entry = SubContextEntry(
                content_hash=content_hash,
                token_ids=token_ids,
                orig_position=orig_position,
                slots=tuple(dest_slots),
                subcontext_id=subcontext_id,
            )
            self._entries[content_hash] = entry
            self._refcount[content_hash] = 0
            self._ring_span[content_hash] = (ring_positions[0], length)
            for p in ring_positions:
                self._ring_owner[p] = content_hash
            probe = probe_hash(token_ids, 0)
            if probe is not None:
                self._by_probe.setdefault(probe, []).append(content_hash)
                self._hash_to_probe[content_hash] = probe
            return dest_slots

    def lookup(self, content_hash: str) -> Optional[SubContextEntry]:
        with self._lock:
            entry = self._entries.get(content_hash)
            if entry is not None:
                self._entries.move_to_end(content_hash)
            return entry

    def candidates_for_probe(self, probe: str) -> List[SubContextEntry]:
        with self._lock:
            hashes = self._by_probe.get(probe, ())
            return [self._entries[h] for h in hashes if h in self._entries]

    def inc_ref(self, content_hash: str) -> None:
        with self._lock:
            self._refcount[content_hash] = self._refcount.get(content_hash, 0) + 1

    def dec_ref(self, content_hash: str) -> None:
        with self._lock:
            if content_hash in self._refcount:
                self._refcount[content_hash] = max(0, self._refcount[content_hash] - 1)

    def _reserve_ring_range_locked(self, length: int) -> Optional[List[int]]:
        """First-fit search starting at the write cursor: find `length`
        contiguous ring positions with no *referenced* (in-flight) occupant,
        evicting whatever unreferenced entries are in the way, skip past a
        referenced one and keep looking, wrap at the buffer end. Bounded to
        one full pass over the ring so a pool saturated with in-flight
        entries fails fast instead of spinning.
        """
        capacity = len(self._slots)
        pos = self._cursor
        probed = 0
        while probed <= capacity:
            if pos + length > capacity:
                pos = 0
            blocked_at = None
            for p in range(pos, pos + length):
                owner = self._ring_owner[p]
                if owner is not None and self._refcount.get(owner, 0) > 0:
                    blocked_at = p
                    break
            if blocked_at is None:
                evicted = set()
                for p in range(pos, pos + length):
                    owner = self._ring_owner[p]
                    if owner is not None and owner not in evicted:
                        self._evict_entry_locked(owner)
                        evicted.add(owner)
                self._cursor = (pos + length) % capacity
                return list(range(pos, pos + length))
            probed += (blocked_at - pos) + 1
            pos = blocked_at + 1
        return None

    def _evict_entry_locked(self, content_hash: str) -> None:
        self._entries.pop(content_hash, None)
        self._refcount.pop(content_hash, None)
        self._ring_span.pop(content_hash, None)
        probe = self._hash_to_probe.pop(content_hash, None)
        if probe is not None and probe in self._by_probe:
            self._by_probe[probe].remove(content_hash)
            if not self._by_probe[probe]:
                del self._by_probe[probe]
