"""
Unit tests for the sub-context KV cache (mem_cache/subcontext/,
layers/rotary_embedding/reposition.py).

Pure CPU/tensor logic -- no model, no attention backend -- so these run on
any runner. Each case pins a derived property or a bookkeeping invariant
per .claude/rules/unit-test-admission.md, not a restatement of the
implementation.
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

import unittest

import torch

from sglang.srt.layers.rotary_embedding.reposition import reposition_key
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.mem_cache.subcontext.deviation_recompute import (
    plan_boundary_recompute_zones,
)
from sglang.srt.mem_cache.subcontext.extend_plan import (
    build_batch_subcontext_plan,
    plan_request_extend,
    reused_ranges_within,
    source_slot_for_position,
    surviving_offsets,
)
from sglang.srt.mem_cache.subcontext.kv_materialize import (
    attention_backend_supports_subcontext_reuse,
    materialize_reused_kv,
)
from sglang.srt.mem_cache.subcontext.subcontext_index import SubContextIndex, hash_token_span
from sglang.srt.mem_cache.subcontext.subcontext_scanner import scan
from sglang.srt.mem_cache.subcontext.subcontext_types import SubContextEntry, SubContextTag


def _make_entry(token_ids, orig_position=0, subcontext_id=None):
    """Builds a standalone SubContextEntry for tests that consume one
    directly (SubContextMatch/RecomputePlan), without going through a
    SubContextIndex's ring-buffer slot reservation."""
    token_ids = tuple(token_ids)
    return SubContextEntry(
        content_hash=hash_token_span(token_ids),
        token_ids=token_ids,
        orig_position=orig_position,
        slots=tuple(range(len(token_ids))),
        subcontext_id=subcontext_id,
    )


def _new_index(capacity: int = 1000) -> SubContextIndex:
    index = SubContextIndex()
    index.bind_slots(range(capacity))
    return index


def _register(index, token_ids, orig_position=0, subcontext_id=None):
    """Registers via the real SubContextIndex.register() API (ring-buffer
    slot reservation + copy target) and returns the resulting entry."""
    token_ids = tuple(token_ids)
    index.register(
        token_ids=token_ids, orig_position=orig_position, subcontext_id=subcontext_id
    )
    return index.lookup(hash_token_span(token_ids))


def _no_recompute(match):
    """Fixture helper for tests that need *some* RecomputePlan but aren't
    testing the BRZ policy itself (extend-plan/stitching mechanics). Routes
    through the real plan_boundary_recompute_zones with k=0 rather than
    reinventing "empty recompute_offsets" locally, so it can't drift from
    what k=0 actually produces."""
    return plan_boundary_recompute_zones([match], k=0)[0]


class TestRepositionMath(unittest.TestCase):
    """Rotation composition is the load-bearing property that makes reuse
    at a different position mathematically valid rather than a heuristic
    approximation; a regression here silently corrupts every reused chunk's
    attention scores."""

    def _check(self, is_neox_style: bool):
        torch.manual_seed(0)
        num_tokens, num_heads, head_dim = 13, 4, 64
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        raw_k = torch.randn(num_tokens, num_heads, head_dim)
        orig_pos = torch.randint(0, 500, (num_tokens,)).float()
        new_pos = orig_pos + torch.randint(-50, 800, (num_tokens,)).float()

        def rope_at(positions):
            freqs = torch.einsum("i,j->ij", positions, inv_freq)
            return freqs.cos(), freqs.sin()

        k_cached = apply_rotary_emb(raw_k, *rope_at(orig_pos), is_neox_style)
        k_repositioned = reposition_key(
            k_cached,
            delta_positions=new_pos - orig_pos,
            inv_freq=inv_freq,
            is_neox_style=is_neox_style,
        )
        k_expected = apply_rotary_emb(raw_k, *rope_at(new_pos), is_neox_style)
        torch.testing.assert_close(k_repositioned, k_expected, atol=1e-4, rtol=1e-4)

    def test_neox_style_composition(self):
        self._check(is_neox_style=True)

    def test_gptj_style_composition(self):
        self._check(is_neox_style=False)

    def test_zero_delta_is_identity(self):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
        k = torch.randn(5, 2, 64)
        out = reposition_key(
            k, delta_positions=torch.zeros(5), inv_freq=inv_freq, is_neox_style=True
        )
        torch.testing.assert_close(out, k, atol=1e-5, rtol=1e-5)


class _FakePool:
    """Minimal KVCache double: one [pool_size, num_kv_heads, head_dim]
    tensor per layer, exposing exactly the two accessors
    materialize_reused_kv uses."""

    def __init__(self, num_layers, pool_size, num_kv_heads, head_dim):
        self.k = [torch.randn(pool_size, num_kv_heads, head_dim) for _ in range(num_layers)]
        self.v = [torch.randn(pool_size, num_kv_heads, head_dim) for _ in range(num_layers)]

    def get_key_buffer(self, layer_id):
        return self.k[layer_id]

    def get_value_buffer(self, layer_id):
        return self.v[layer_id]


class TestMaterializeReusedKv(unittest.TestCase):
    def test_materialized_k_matches_direct_rotation_and_v_is_copied(self):
        """End-to-end (minus the real model): the slots a request lands on
        after materialization must hold exactly what full recomputation at
        the new position would have produced for K, and an untouched copy
        for V -- the whole point of the mechanism is that this substitutes
        for recomputation losslessly."""
        torch.manual_seed(0)
        num_layers, pool_size, num_heads, head_dim = 3, 64, 4, 32
        pool = _FakePool(num_layers, pool_size, num_heads, head_dim)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))

        source_slots = torch.tensor([5, 6, 7])
        dest_slots = torch.tensor([40, 41, 42])
        orig_positions = torch.tensor([5.0, 6.0, 7.0])
        new_positions = torch.tensor([100.0, 101.0, 102.0])
        delta = new_positions - orig_positions

        expected_k = [pool.get_key_buffer(l).index_select(0, source_slots).clone() for l in range(num_layers)]
        expected_v = [pool.get_value_buffer(l).index_select(0, source_slots).clone() for l in range(num_layers)]

        materialize_reused_kv(
            token_to_kv_pool=pool,
            layer_ids=range(num_layers),
            source_slots=source_slots,
            dest_slots=dest_slots,
            delta_positions=delta,
            inv_freq=inv_freq,
            is_neox_style=True,
            rotary_dim=head_dim,
        )

        for l in range(num_layers):
            got_k = pool.get_key_buffer(l).index_select(0, dest_slots)
            want_k = reposition_key(
                expected_k[l], delta_positions=delta, inv_freq=inv_freq, is_neox_style=True
            )
            torch.testing.assert_close(got_k, want_k, atol=1e-5, rtol=1e-5)

            got_v = pool.get_value_buffer(l).index_select(0, dest_slots)
            torch.testing.assert_close(got_v, expected_v[l])

    def test_partial_rotary_leaves_pass_through_dims_untouched(self):
        """Only the leading rotary_dim channels of K may change; the
        pass-through tail must be a byte-for-byte copy, same as V."""
        torch.manual_seed(1)
        head_dim, rotary_dim = 32, 16
        pool = _FakePool(1, 16, 2, head_dim)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))

        source_slots = torch.tensor([0])
        dest_slots = torch.tensor([10])
        expected_pass_through = pool.get_key_buffer(0)[0, :, rotary_dim:].clone()

        materialize_reused_kv(
            token_to_kv_pool=pool,
            layer_ids=[0],
            source_slots=source_slots,
            dest_slots=dest_slots,
            delta_positions=torch.tensor([7.0]),
            inv_freq=inv_freq,
            is_neox_style=True,
            rotary_dim=rotary_dim,
        )

        got_pass_through = pool.get_key_buffer(0)[10, :, rotary_dim:]
        torch.testing.assert_close(got_pass_through, expected_pass_through)


class TestMultiChunkStitching(unittest.TestCase):
    """End-to-end (minus the real model): one request whose extend range
    reuses KV from *two independently registered* sub-context chunks
    (different original positions, different physical source slots --
    "different radix trees" of KV, in the sense that each was computed by
    a different earlier request), stitched together with the tokens that
    still need a real forward pass. Every earlier test in this file
    exercises one match at a time; this is the only one that proves two
    simultaneous chunks don't get cross-wired -- an indexing bug in
    build_batch_subcontext_plan's per-request loop (e.g. reusing the first
    match's source slots for the second) would silently corrupt attention
    for the second chunk while every single-match test kept passing.

    Also pins down that position-independent coding (the RoPE delta
    rotation) actually runs per chunk after stitching, rather than being
    skipped or shared across chunks: verified against two independent
    negative controls (repositioning turned into a no-op; one chunk's
    delta silently swapped for the other's), both of which the assertions
    here catch.
    """

    def test_two_independent_chunks_stitch_without_cross_contamination(self):
        torch.manual_seed(0)
        num_layers, pool_size, num_heads, head_dim = 2, 2048, 2, 16
        pool = _FakePool(num_layers, pool_size, num_heads, head_dim)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Request layout: prefix_len=5 (radix-matched, out of scope here),
        # extend range covers absolute positions [5, 25). Two entries with
        # disjoint, identifiable source slots -- "different radix trees" of
        # KV, each computed by a different earlier request.
        from sglang.srt.mem_cache.subcontext.subcontext_types import SubContextMatch

        prefix_len, extend_len = 5, 20
        entry_a = SubContextEntry(
            content_hash="a", token_ids=tuple(range(0, 5)), orig_position=50,
            slots=(10, 11, 12, 13, 14),
        )
        entry_b = SubContextEntry(
            content_hash="b", token_ids=tuple(range(0, 4)), orig_position=200,
            slots=(90, 91, 92, 93),
        )
        match_a = SubContextMatch(entry=entry_a, query_start=7, query_end=12, source="scanned")
        match_b = SubContextMatch(entry=entry_b, query_start=16, query_end=20, source="scanned")

        plan = plan_request_extend(
            [_no_recompute(match_a), _no_recompute(match_b)],
            prefix_len=prefix_len,
            extend_len=extend_len,
        )
        # out_cache_loc as alloc_for_extend would hand back: one physical
        # slot per logical position in the extend range, contiguous.
        out_cache_loc = torch.arange(2000, 2000 + extend_len)
        batch_plan = build_batch_subcontext_plan(
            request_plans=[plan],
            prefix_lens=[prefix_len],
            extend_lens=[extend_len],
            out_cache_loc=out_cache_loc,
        )

        expected_k_a = [pool.get_key_buffer(l)[list(entry_a.slots)].clone() for l in range(num_layers)]
        expected_k_b = [pool.get_key_buffer(l)[list(entry_b.slots)].clone() for l in range(num_layers)]
        expected_v_a = [pool.get_value_buffer(l)[list(entry_a.slots)].clone() for l in range(num_layers)]
        expected_v_b = [pool.get_value_buffer(l)[list(entry_b.slots)].clone() for l in range(num_layers)]
        untouched_before = [pool.get_key_buffer(l).clone() for l in range(num_layers)]

        materialize_reused_kv(
            token_to_kv_pool=pool,
            layer_ids=range(num_layers),
            source_slots=batch_plan.materialize_source_slots,
            dest_slots=batch_plan.materialize_dest_slots,
            delta_positions=batch_plan.materialize_delta_positions,
            inv_freq=inv_freq,
            is_neox_style=True,
            rotary_dim=head_dim,
        )

        # dest slots = out_cache_loc at match_a/b's local offsets within
        # the extend range: match_a -> [2,7), match_b -> [11,15).
        dest_a = out_cache_loc[2:7]
        dest_b = out_cache_loc[11:15]
        delta_a = float(match_a.delta_position)
        delta_b = float(match_b.delta_position)

        def rotate(expected, delta, count):
            return reposition_key(
                expected,
                delta_positions=torch.full((count,), delta),
                inv_freq=inv_freq,
                is_neox_style=True,
            )

        for l in range(num_layers):
            got_k_a = pool.get_key_buffer(l)[dest_a]
            want_k_a = rotate(expected_k_a[l], delta_a, 5)
            torch.testing.assert_close(got_k_a, want_k_a, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(pool.get_value_buffer(l)[dest_a], expected_v_a[l])

            got_k_b = pool.get_key_buffer(l)[dest_b]
            want_k_b = rotate(expected_k_b[l], delta_b, 4)
            torch.testing.assert_close(got_k_b, want_k_b, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(pool.get_value_buffer(l)[dest_b], expected_v_b[l])

            # Position-independent coding must actually run per chunk, not
            # get skipped or shared across chunks once multiple are stitched
            # into one row: a chunk's stored K must differ from its raw
            # un-rotated source (rotation had a real, non-trivial effect --
            # delta_a/b are large enough that a no-op bug can't hide here),
            # and must differ from what the *other* chunk's delta would have
            # produced (rules out one chunk silently inheriting the other's
            # position shift).
            self.assertFalse(torch.allclose(got_k_a, expected_k_a[l], atol=1e-5))
            self.assertFalse(
                torch.allclose(got_k_a, rotate(expected_k_a[l], delta_b, 5), atol=1e-5)
            )
            self.assertFalse(torch.allclose(got_k_b, expected_k_b[l], atol=1e-5))
            self.assertFalse(
                torch.allclose(got_k_b, rotate(expected_k_b[l], delta_a, 4), atol=1e-5)
            )

            # Surviving (glue) positions were never touched by materialization
            # -- the live forward still owns them.
            survive_slots = out_cache_loc[list(plan.surviving_local)]
            torch.testing.assert_close(
                pool.get_key_buffer(l)[survive_slots], untouched_before[l][survive_slots]
            )


class TestExtendPlan(unittest.TestCase):
    def _match(self, start, end, orig_position=0):
        from sglang.srt.mem_cache.subcontext.subcontext_types import SubContextMatch

        entry = _make_entry(range(0, end - start), orig_position=orig_position)
        return SubContextMatch(entry=entry, query_start=start, query_end=end, source="scanned")

    def test_match_split_across_chunk_boundary_is_dropped(self):
        """A match only partly inside this round's extend range must be
        fully dropped (computed normally), not partially materialized --
        chunked prefill would otherwise reuse KV for tokens this chunk
        never actually allocated slots for."""
        plan = _no_recompute(self._match(10, 30))  # extends past extend_end=20
        ranges = reused_ranges_within([plan], extend_start=0, extend_end=20)
        self.assertEqual(ranges, [])

    def test_fully_covered_match_is_kept(self):
        plan = _no_recompute(self._match(10, 20))
        ranges = reused_ranges_within([plan], extend_start=0, extend_end=20)
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0][:2], (10, 20))

    def test_surviving_offsets_is_the_complement_of_reused_ranges(self):
        """Derived property: surviving + reused must partition
        [extend_start, extend_end) exactly, with no overlap and no gap --
        get this wrong and a token either never gets computed (garbage
        output) or gets computed twice (wasted, but silently so)."""
        m = self._match(2, 5)
        reused = [(2, 5, m), (8, 9, m)]
        survive = surviving_offsets(reused, extend_start=0, extend_end=10)
        self.assertEqual(survive, [0, 1, 5, 6, 7, 9])

        reused_flat = set()
        for s, e, _m in reused:
            reused_flat.update(range(s, e))
        self.assertEqual(set(survive) | reused_flat, set(range(0, 10)))
        self.assertEqual(set(survive) & reused_flat, set())

    def test_source_slot_for_position_indexes_from_match_start(self):
        m = self._match(10, 15)  # entry.slots = (0, 1, 2, 3, 4)
        self.assertEqual(source_slot_for_position(m, 10), 0)
        self.assertEqual(source_slot_for_position(m, 13), 3)

    def test_plan_request_extend_partitions_the_full_local_range(self):
        """Derived property: surviving_local + reused_local must equal
        exactly {0, ..., extend_len-1} with no overlap -- this is what
        lets the caller split one flat out_cache_loc slice by these index
        sets and use every physical slot exactly once."""
        prefix_len, extend_len = 100, 20
        # Match covers absolute [105, 112) i.e. local [5, 12); orig_position=0
        # so delta_position = 105 - 0 = 105.
        entry = _make_entry(range(0, 7), orig_position=0)
        from sglang.srt.mem_cache.subcontext.subcontext_types import SubContextMatch

        match = SubContextMatch(entry=entry, query_start=105, query_end=112, source="scanned")
        plan = plan_request_extend(
            [_no_recompute(match)], prefix_len=prefix_len, extend_len=extend_len
        )

        self.assertEqual(plan.reused_local, tuple(range(5, 12)))
        self.assertEqual(
            set(plan.surviving_local) | set(plan.reused_local), set(range(extend_len))
        )
        self.assertEqual(set(plan.surviving_local) & set(plan.reused_local), set())
        self.assertEqual(plan.reused_source_slots, entry.slots)
        self.assertEqual(plan.reused_delta_positions, (105.0,) * 7)
        self.assertEqual(
            plan.surviving_absolute, tuple(prefix_len + o for o in plan.surviving_local)
        )

    def test_plan_request_extend_with_no_matches_keeps_everything_surviving(self):
        prefix_len, extend_len = 5, 10
        plan = plan_request_extend([], prefix_len=prefix_len, extend_len=extend_len)
        self.assertEqual(plan.surviving_local, tuple(range(extend_len)))
        self.assertEqual(plan.reused_local, ())

    def test_build_batch_subcontext_plan_mixed_batch(self):
        """A request with no plan (feature off / ineligible / no match)
        must come through byte-identical to today's contiguous slice; a
        request with a plan must have its out_cache_loc narrowed to
        exactly its surviving slots, its extend_len shrunk to match, and
        its reused slots + deltas collected for the materializer -- this
        is the exact transform prepare_for_extend applies to the whole
        batch's already-allocated out_cache_loc."""
        from sglang.srt.mem_cache.subcontext.extend_plan import RequestSubcontextPlan

        out_cache_loc = torch.tensor([100, 101, 102, 200, 201, 202, 203, 204])
        prefix_lens = [0, 50]
        extend_lens = [3, 5]
        req1_plan = RequestSubcontextPlan(
            surviving_absolute=(50, 53, 54),
            surviving_local=(0, 3, 4),
            reused_local=(1, 2),
            reused_source_slots=(10, 11),
            reused_delta_positions=(40.0, 40.0),
        )

        result = build_batch_subcontext_plan(
            request_plans=[None, req1_plan],
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
            out_cache_loc=out_cache_loc,
        )

        torch.testing.assert_close(
            result.out_cache_loc, torch.tensor([100, 101, 102, 200, 203, 204])
        )
        self.assertEqual(result.extend_lens, [3, 3])
        torch.testing.assert_close(result.positions, torch.tensor([0, 1, 2, 50, 53, 54]))
        torch.testing.assert_close(result.materialize_dest_slots, torch.tensor([201, 202]))
        torch.testing.assert_close(result.materialize_source_slots, torch.tensor([10, 11]))
        torch.testing.assert_close(
            result.materialize_delta_positions, torch.tensor([40.0, 40.0])
        )

    def test_build_batch_subcontext_plan_all_passthrough_matches_original(self):
        """No plans active (the common case: feature off, or on but no
        request in this batch matched anything) must reproduce the
        original out_cache_loc and a plain arange positions tensor
        exactly -- this is the zero-overhead-when-idle guarantee."""
        out_cache_loc = torch.tensor([5, 6, 7, 8, 9])
        result = build_batch_subcontext_plan(
            request_plans=[None, None],
            prefix_lens=[0, 3],
            extend_lens=[3, 2],
            out_cache_loc=out_cache_loc,
        )
        torch.testing.assert_close(result.out_cache_loc, out_cache_loc)
        self.assertEqual(result.extend_lens, [3, 2])
        torch.testing.assert_close(result.positions, torch.tensor([0, 1, 2, 3, 4]))
        self.assertEqual(result.materialize_source_slots.numel(), 0)


class TestAttentionBackendSupportsSubcontextReuse(unittest.TestCase):
    """Regression: FlashInfer's default ragged prefill fast path sizes the
    KV context it reads off extend_prefix_lens alone (the literal radix
    prefix), never the materialized reused span sitting in the gap between
    it and the surviving suffix -- so a reused chunk's KV would silently
    never be attended to. FlashAttention's extend path has no such split
    (always sizes off the full seq_lens). This must stay a strict allowlist:
    an unaudited or unrecognized backend defaults to unsupported, not to
    trusted."""

    def test_fa3_is_supported(self):
        self.assertTrue(attention_backend_supports_subcontext_reuse("fa3"))

    def test_flashinfer_requires_paged_mode(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_FLASHINFER_USE_PAGED.override(False):
            self.assertFalse(attention_backend_supports_subcontext_reuse("flashinfer"))
        with envs.SGLANG_FLASHINFER_USE_PAGED.override(True):
            self.assertTrue(attention_backend_supports_subcontext_reuse("flashinfer"))

    def test_unrecognized_backend_defaults_to_unsupported(self):
        self.assertFalse(attention_backend_supports_subcontext_reuse("triton"))
        self.assertFalse(attention_backend_supports_subcontext_reuse("torch_native"))


class TestSubContextIndex(unittest.TestCase):
    def test_referenced_entry_survives_eviction_pressure(self):
        """A chunk locked by an in-flight request must not be evicted even
        under ring-buffer pressure -- the same contract RadixCache's
        inc_lock_ref/dec_lock_ref gives tree nodes. A 16-slot pool holding
        two 8-token entries is exactly full; registering a third must skip
        past the referenced one and evict only the unreferenced one."""
        index = SubContextIndex()
        index.bind_slots(range(16))
        e1_hash = hash_token_span(tuple(range(0, 8)))
        e2_hash = hash_token_span(tuple(range(100, 108)))
        e3_hash = hash_token_span(tuple(range(200, 208)))

        index.register(token_ids=range(0, 8), orig_position=0)
        index.register(token_ids=range(100, 108), orig_position=100)
        index.inc_ref(e1_hash)

        index.register(token_ids=range(200, 208), orig_position=200)

        self.assertIsNotNone(index.lookup(e1_hash))
        self.assertIsNone(index.lookup(e2_hash))
        self.assertIsNotNone(index.lookup(e3_hash))

    def test_registering_a_span_longer_than_the_pool_is_a_no_op(self):
        """A span that could never fit must be rejected cleanly (no
        partial write, no crash), not treated as a 0-length reservation."""
        index = SubContextIndex()
        index.bind_slots(range(4))
        dest = index.register(token_ids=range(0, 8), orig_position=0)
        self.assertIsNone(dest)
        self.assertEqual(len(index), 0)

    def test_registration_wraps_to_the_start_when_the_tail_is_too_small(self):
        """A 6-token entry can't fit in the last 4 slots of a 10-slot pool
        (positions [6,10)); registration must wrap to the start rather
        than silently under-allocate or write past the pool end -- the
        prior occupant of [0,6) gets evicted to make room."""
        index = SubContextIndex()
        index.bind_slots(range(10))
        first_hash = hash_token_span(tuple(range(0, 6)))
        index.register(token_ids=range(0, 6), orig_position=0)  # -> ring[0:6]

        dest = index.register(token_ids=range(100, 106), orig_position=100)
        self.assertEqual(dest, list(range(0, 6)))  # wrapped, not [6, 12)
        self.assertIsNone(index.lookup(first_hash))

    def test_reregistering_the_same_content_is_a_no_op(self):
        """Dedup: registering identical content twice must not consume a
        second ring-buffer reservation (bookkeeping-only op)."""
        index = SubContextIndex()
        index.bind_slots(range(16))
        first = index.register(token_ids=range(0, 8), orig_position=0)
        second = index.register(token_ids=range(0, 8), orig_position=0)
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(index), 1)


class TestScanner(unittest.TestCase):
    def test_prefers_longest_verified_match(self):
        """Two registered chunks share a common prefix; scanning must pick
        the longer one so a short chunk never shadows a bigger reuse
        opportunity it happens to start with."""
        index = _new_index()
        short = _register(index, range(0, 8))
        long = _register(index, list(range(0, 8)) + list(range(50, 58)))

        query = list(range(0, 8)) + list(range(50, 58)) + [999]
        matches = scan(index, query)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].entry.content_hash, long.content_hash)
        self.assertEqual((matches[0].query_start, matches[0].query_end), (0, 16))

    def test_explicit_tag_suppresses_overlapping_auto_scan(self):
        """An explicit tag's span must not also be claimed by the
        automatic scanner (double-covering would double-count reuse and
        corrupt the recompute plan built on top of it)."""
        index = _new_index()
        _register(index, range(0, 8), subcontext_id="sys-prompt")

        query = list(range(0, 8)) + [999]
        tags = [SubContextTag(subcontext_id="sys-prompt", start=0, end=8)]
        matches = scan(index, query, explicit_tags=tags)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].source, "explicit")

    def test_start_excludes_the_prefix_matched_region(self):
        """A chunk that happens to recur inside the region the caller
        already resolved via ordinary prefix caching (before `start`) must
        not be re-reported here -- prefix reuse is strictly cheaper (no
        rotation, zero-copy) and double-claiming it would corrupt the
        reused-token accounting downstream."""
        index = _new_index()
        _register(index, range(0, 8))

        query = list(range(0, 8)) + [1, 2, 3] + list(range(0, 8))
        matches = scan(index, query, start=8)

        self.assertEqual(len(matches), 1)
        self.assertEqual((matches[0].query_start, matches[0].query_end), (11, 19))

    def test_match_never_covers_the_requests_last_token(self):
        """Regression: a match reaching the request's final token would
        leave prepare_for_extend with zero surviving tokens for that
        request -- no forward pass, no logits, no next-token sample.
        _compute_max_prefix_len caps the ordinary prefix match at
        input_len - 1 for exactly this reason; scan's `end` must give the
        same guarantee for subcontext matches."""
        index = _new_index()
        _register(index, range(0, 9))

        query = list(range(0, 9))  # a 9-token match would exactly cover this whole query
        matches = scan(index, query, end=len(query) - 1)
        self.assertEqual(matches, [])

        # One token shorter (end left open) does match, proving `end` is
        # actually the reason for the miss above, not something else.
        self.assertEqual(len(scan(index, query)), 1)

    def test_no_false_match_on_probe_collision_without_full_verify(self):
        """A candidate sharing only the probe-hash prefix but diverging
        later must be rejected -- the probe is a filter, not proof."""
        index = _new_index()
        _register(index, list(range(0, 8)) + [12345])

        query = list(range(0, 8)) + [99999]  # same first 8 tokens, then diverges
        matches = scan(index, query)
        self.assertEqual(matches, [])


class TestBoundaryRecomputeZone(unittest.TestCase):
    """BRZ(A, B) = Last_k(A) union First_k(B) for adjacent segments A, B.
    plan_boundary_recompute_zones takes the full, query-ordered match list
    so it can tell "adjacent" (no glue between two matches) from "isolated"
    (glue, prefix, or nothing on one side) -- these tests pin down that
    distinction, since getting it wrong either recomputes tokens no
    boundary needs (wasted compute) or silently skips a real stitch."""

    def _match(self, entry, start, end):
        from sglang.srt.mem_cache.subcontext.subcontext_types import SubContextMatch

        return SubContextMatch(entry=entry, query_start=start, query_end=end, source="scanned")

    def test_isolated_match_gets_only_leading_first_k(self):
        """A match with nothing stitched after it (end of range, or glue
        follows) must recompute only its own leading k tokens -- First_k(B)
        -- never its trailing edge, since nothing-empty-handed to repair."""
        entry = _make_entry(range(0, 10), orig_position=0)
        match = self._match(entry, 20, 30)
        (plan,) = plan_boundary_recompute_zones([match], k=3)

        self.assertEqual(plan.recompute_offsets, (0, 1, 2))
        self.assertEqual(plan.reused_ranges(), ((23, 30),))

    def test_stitched_pair_gets_last_k_and_first_k_on_each_side(self):
        """Two matches with zero gap between them (match_a.query_end ==
        match_b.query_start) is exactly BRZ's "adjacent segments" case:
        match_a's trailing k tokens (Last_k(A)) and match_b's leading k
        tokens (First_k(B)) both get recomputed."""
        entry_a = _make_entry(range(0, 10), orig_position=0)
        entry_b = _make_entry(range(0, 10), orig_position=100)
        match_a = self._match(entry_a, 20, 30)
        match_b = self._match(entry_b, 30, 40)  # starts exactly where A ends

        plan_a, plan_b = plan_boundary_recompute_zones([match_a, match_b], k=3)

        # A: leading 3 (First_k(A), always) + trailing 3 (Last_k(A), stitched to B)
        self.assertEqual(plan_a.recompute_offsets, (0, 1, 2, 7, 8, 9))
        self.assertEqual(plan_a.reused_ranges(), ((23, 27),))
        # B: only its own leading 3 -- nothing is stitched after B here.
        self.assertEqual(plan_b.recompute_offsets, (0, 1, 2))
        self.assertEqual(plan_b.reused_ranges(), ((33, 40),))

    def test_glue_between_matches_suppresses_last_k(self):
        """The same two matches, but with a 1-token gap (glue) between
        them, must NOT trigger match_a's Last_k -- glue is already
        freshly computed against match_a's true tail, so there is no
        second boundary to repair on A's side."""
        entry_a = _make_entry(range(0, 10), orig_position=0)
        entry_b = _make_entry(range(0, 10), orig_position=100)
        match_a = self._match(entry_a, 20, 30)
        match_b = self._match(entry_b, 31, 41)  # position 30 is glue

        plan_a, plan_b = plan_boundary_recompute_zones([match_a, match_b], k=3)

        self.assertEqual(plan_a.recompute_offsets, (0, 1, 2))  # no trailing 7,8,9
        self.assertEqual(plan_b.recompute_offsets, (0, 1, 2))

    def test_window_larger_than_match_length_is_capped(self):
        """k exceeding the match's own length must cap at the whole match
        -- First_k and Last_k overlapping/exceeding the span must not
        produce out-of-range offsets or a crash."""
        entry = _make_entry(range(0, 5), orig_position=0)
        match = self._match(entry, 0, 5)
        (plan,) = plan_boundary_recompute_zones([match], k=64)

        self.assertEqual(plan.recompute_offsets, (0, 1, 2, 3, 4))
        self.assertEqual(plan.reused_ranges(), ())

    def test_zero_window_reuses_the_whole_match(self):
        entry = _make_entry(range(0, 5), orig_position=0)
        match = self._match(entry, 0, 5)
        (plan,) = plan_boundary_recompute_zones([match], k=0)
        self.assertEqual(plan.recompute_offsets, ())
        self.assertEqual(plan.reused_ranges(), ((0, 5),))

    def test_negative_window_is_rejected(self):
        """k is a token count; negative is meaningless and must fail loud
        rather than silently produce a garbage range() call. (The spec's
        {0, 16, 32, 64} enum is enforced separately, at the config surface
        -- see TestValidBrzWindows below -- since the algorithm itself
        works for any non-negative k.)"""
        entry = _make_entry(range(0, 5), orig_position=0)
        match = self._match(entry, 0, 5)
        with self.assertRaises(AssertionError):
            plan_boundary_recompute_zones([match], k=-1)


class TestValidBrzWindows(unittest.TestCase):
    def test_matches_the_cli_choices(self):
        """Bookkeeping: VALID_BRZ_WINDOWS is what both the CLI's `choices`
        (arg_groups/fields/memory.py) and the startup check
        (Scheduler.maybe_init_subcontext_index) validate against -- drift
        here would let an unsupported k slip past one but not the other."""
        from sglang.srt.mem_cache.subcontext.deviation_recompute import (
            VALID_BRZ_WINDOWS,
        )

        self.assertEqual(VALID_BRZ_WINDOWS, (0, 16, 32, 64))


if __name__ == "__main__":
    unittest.main()
