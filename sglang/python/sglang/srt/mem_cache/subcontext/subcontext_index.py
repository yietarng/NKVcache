"""Content-addressed index of registered sub-context KV chunks.

Unlike ``RadixCache`` (keyed by root-to-node prefix path), this index is
keyed purely by content hash: a sub-context can be looked up regardless of
where it sits in a request's token sequence. See ``subcontext_scanner.py``
for how a new request's tokens are matched against it.
"""

from __future__ import annotations

import hashlib
import struct
import threading
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence

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

    def __init__(self, *, max_entries: int = 100_000):
        self._max_entries = max_entries
        self._lock = threading.Lock()
        # LRU by recency of *lookup*, so hot chunks survive eviction.
        self._entries: "OrderedDict[str, SubContextEntry]" = OrderedDict()
        self._by_probe: Dict[str, List[str]] = {}
        self._refcount: Dict[str, int] = {}
        self._hash_to_probe: Dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def register(self, entry: SubContextEntry) -> None:
        with self._lock:
            if entry.content_hash in self._entries:
                self._entries.move_to_end(entry.content_hash)
                return
            self._evict_locked_if_needed()
            self._entries[entry.content_hash] = entry
            self._refcount.setdefault(entry.content_hash, 0)
            probe = probe_hash(entry.token_ids, 0)
            if probe is not None:
                self._by_probe.setdefault(probe, []).append(entry.content_hash)
                self._hash_to_probe[entry.content_hash] = probe

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

    def _evict_locked_if_needed(self) -> None:
        while len(self._entries) >= self._max_entries:
            evicted = self._pop_lru_unreferenced_locked()
            if evicted is None:
                # Everything is referenced; refuse to grow further rather
                # than evict something in-flight out from under a request.
                return

    def _pop_lru_unreferenced_locked(self) -> Optional[str]:
        for content_hash in self._entries:
            if self._refcount.get(content_hash, 0) == 0:
                del self._entries[content_hash]
                self._refcount.pop(content_hash, None)
                probe = self._hash_to_probe.pop(content_hash, None)
                if probe is not None and probe in self._by_probe:
                    self._by_probe[probe].remove(content_hash)
                    if not self._by_probe[probe]:
                        del self._by_probe[probe]
                return content_hash
        return None

    def evictable_entries(self) -> Iterable[SubContextEntry]:
        with self._lock:
            return [
                e
                for h, e in self._entries.items()
                if self._refcount.get(h, 0) == 0
            ]
