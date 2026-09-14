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
from sglang.srt.mem_cache.subcontext.deviation_recompute import plan_prefix_fraction
from sglang.srt.mem_cache.subcontext.subcontext_index import SubContextIndex, hash_token_span
from sglang.srt.mem_cache.subcontext.subcontext_scanner import scan
from sglang.srt.mem_cache.subcontext.subcontext_types import SubContextEntry, SubContextTag


def _make_entry(token_ids, orig_position=0, subcontext_id=None):
    token_ids = tuple(token_ids)
    return SubContextEntry(
        content_hash=hash_token_span(token_ids),
        token_ids=token_ids,
        orig_position=orig_position,
        slots=tuple(range(len(token_ids))),
        subcontext_id=subcontext_id,
    )


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


class TestSubContextIndex(unittest.TestCase):
    def test_referenced_entry_survives_eviction_pressure(self):
        """A chunk locked by an in-flight request must not be evicted even
        when the index is at capacity -- the same contract RadixCache's
        inc_lock_ref/dec_lock_ref gives tree nodes."""
        index = SubContextIndex(max_entries=2)
        e1 = _make_entry(range(0, 8))
        e2 = _make_entry(range(100, 108))
        index.register(e1)
        index.register(e2)
        index.inc_ref(e1.content_hash)

        e3 = _make_entry(range(200, 208))
        index.register(e3)  # forces an eviction: e1 is locked, e2 is not

        self.assertIsNotNone(index.lookup(e1.content_hash))
        self.assertIsNone(index.lookup(e2.content_hash))


class TestScanner(unittest.TestCase):
    def test_prefers_longest_verified_match(self):
        """Two registered chunks share a common prefix; scanning must pick
        the longer one so a short chunk never shadows a bigger reuse
        opportunity it happens to start with."""
        index = SubContextIndex()
        short = _make_entry(range(0, 8))
        long = _make_entry(list(range(0, 8)) + list(range(50, 58)))
        index.register(short)
        index.register(long)

        query = list(range(0, 8)) + list(range(50, 58)) + [999]
        matches = scan(index, query)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].entry.content_hash, long.content_hash)
        self.assertEqual((matches[0].query_start, matches[0].query_end), (0, 16))

    def test_explicit_tag_suppresses_overlapping_auto_scan(self):
        """An explicit tag's span must not also be claimed by the
        automatic scanner (double-covering would double-count reuse and
        corrupt the recompute plan built on top of it)."""
        index = SubContextIndex()
        entry = _make_entry(range(0, 8), subcontext_id="sys-prompt")
        index.register(entry)

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
        index = SubContextIndex()
        entry = _make_entry(range(0, 8))
        index.register(entry)

        query = list(range(0, 8)) + [1, 2, 3] + list(range(0, 8))
        matches = scan(index, query, start=8)

        self.assertEqual(len(matches), 1)
        self.assertEqual((matches[0].query_start, matches[0].query_end), (11, 19))

    def test_no_false_match_on_probe_collision_without_full_verify(self):
        """A candidate sharing only the probe-hash prefix but diverging
        later must be rejected -- the probe is a filter, not proof."""
        index = SubContextIndex()
        entry = _make_entry(list(range(0, 8)) + [12345])
        index.register(entry)

        query = list(range(0, 8)) + [99999]  # same first 8 tokens, then diverges
        matches = scan(index, query)
        self.assertEqual(matches, [])


class TestRecomputePlan(unittest.TestCase):
    def test_reused_ranges_excludes_recomputed_offsets(self):
        """Boundary math: reused_ranges() must be exactly the match span
        minus the recomputed offsets, as disjoint contiguous pieces --
        get this wrong and reused KV silently overlaps recomputed KV."""
        index = SubContextIndex()
        entry = _make_entry(range(0, 10), orig_position=0)
        from sglang.srt.mem_cache.subcontext.subcontext_types import SubContextMatch

        match = SubContextMatch(entry=entry, query_start=20, query_end=30, source="scanned")
        plan = plan_prefix_fraction(match, recompute_ratio=0.3)  # ceil(10*0.3)=3

        self.assertEqual(plan.recompute_offsets, (0, 1, 2))
        self.assertEqual(plan.reused_ranges(), ((23, 30),))

    def test_zero_ratio_reuses_the_whole_match(self):
        index = SubContextIndex()
        entry = _make_entry(range(0, 5), orig_position=0)
        from sglang.srt.mem_cache.subcontext.subcontext_types import SubContextMatch

        match = SubContextMatch(entry=entry, query_start=0, query_end=5, source="scanned")
        plan = plan_prefix_fraction(match, recompute_ratio=0.0)
        self.assertEqual(plan.reused_ranges(), ((0, 5),))


if __name__ == "__main__":
    unittest.main()
