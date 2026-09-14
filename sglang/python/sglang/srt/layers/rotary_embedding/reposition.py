"""Position-independent KV reuse: re-encode a cached K for a new position.

RoPE rotates K by a matrix that depends only on absolute position, and that
rotation composes additively: ``Rot(p_new) = Rot(p_new - p_old) @ Rot(p_old)``.
So a K vector cached at ``p_old`` (i.e. ``Rot(p_old) @ k_raw``) can be moved to
``p_new`` by applying one more rotation of ``delta = p_new - p_old`` on top of
the stored value, without ever recovering ``k_raw`` or rerunning the K
projection. V carries no positional encoding and needs no correction.

This is the mechanism behind sub-context KV reuse (see
``mem_cache/subcontext/``): a sub-context's K/V, once computed, can be
grafted into a new prompt at a different offset by rotating K by the offset
delta instead of recomputing it. It does not correct for the sub-context's
KV having originally attended to different neighboring content -- that is
what ``mem_cache/subcontext/deviation_recompute.py`` is for.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb


def compute_delta_cos_sin(
    *,
    inv_freq: torch.Tensor,
    delta_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of ``delta_positions * inv_freq``, computed directly instead of
    indexed from a cache -- deltas can be negative (a chunk reused earlier
    than it was first seen), which a position-indexed cache can't serve."""
    freqs = torch.einsum(
        "i,j->ij", delta_positions.to(inv_freq.dtype), inv_freq
    )
    return freqs.cos(), freqs.sin()


def reposition_key(
    cached_key: torch.Tensor,
    *,
    delta_positions: torch.Tensor,
    inv_freq: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """Re-encode a cached, already-RoPE'd K for a new absolute position.

    Args:
        cached_key: ``[num_tokens, num_kv_heads, rotary_dim]``. Pass only the
            rotary slice for partial-rotary models (e.g. ``k[..., :rotary_dim]``)
            -- the same slicing the model's own RotaryEmbedding.forward uses --
            and graft the untouched pass-through dims back in the caller.
        delta_positions: ``[num_tokens]``, ``new_position - orig_position``
            per token.
        inv_freq: the rotary embedding's own inverse-frequency buffer, so
            the delta rotation uses the exact same base/scaling as the
            original encoding.
        is_neox_style: must match the model's rotary embedding style.
    """
    cos, sin = compute_delta_cos_sin(inv_freq=inv_freq, delta_positions=delta_positions)
    return apply_rotary_emb(cached_key, cos, sin, is_neox_style)
