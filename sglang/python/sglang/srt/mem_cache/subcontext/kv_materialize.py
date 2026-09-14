"""Materialize reused sub-context KV into a request's freshly allocated slots.

Runs once per batch, before the model forward: for every layer, gather the
source chunk's K/V at its stored slots, reposition K for the new offset
(``layers/rotary_embedding/reposition.py``), and write both into the
requesting batch's newly allocated destination slots. Pure data movement --
no model forward involved -- which is what lets it sit ahead of
``ModelRunner.forward`` instead of inside it (see
``managers/tp_worker.py: forward_batch_generation`` for the call site and
``docs/docs/advanced_features/subcontext_kv_cache.mdx`` for why this is
sufficient: prefix caching already skips the forward pass entirely for
cached tokens, this generalizes that to non-leading spans).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.reposition import reposition_key

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache
    from sglang.srt.mem_cache.subcontext.subcontext_types import (
        SubcontextMaterializePlan,
    )


def find_rotary_embedding(model: torch.nn.Module) -> Optional[RotaryEmbedding]:
    """The model's shared RotaryEmbedding module, if it has exactly the kind
    this MVP supports: one instance reused by every layer (true of a dense,
    standard-RoPE model). Returns None for anything else -- multiple
    distinct instances (e.g. a model mixing global/local RoPE bases) or
    none at all (NoPE, ALiBi) -- so the caller can skip materialization
    rather than guess which one applies.
    """
    found = None
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            if found is not None and module is not found:
                return None
            found = module
    return found


def materialize_reused_kv(
    *,
    token_to_kv_pool: "KVCache",
    layer_ids: Sequence[int],
    source_slots: torch.Tensor,
    dest_slots: torch.Tensor,
    delta_positions: torch.Tensor,
    inv_freq: torch.Tensor,
    is_neox_style: bool,
    rotary_dim: int,
) -> None:
    """Copy+reposition K, copy V, for every layer, for one batch's worth of
    reused sub-context tokens in one shot.

    Args:
        source_slots, dest_slots, delta_positions: ``[num_reused_tokens]``,
            aligned index-for-index across every layer.
        rotary_dim: only the leading ``rotary_dim`` channels of K get
            rotated; the remaining (partial-rotary) channels are copied
            unchanged along with V.
    """
    if source_slots.numel() == 0:
        return
    delta_positions = delta_positions.to(dtype=torch.float32)

    for layer_id in layer_ids:
        k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
        v_buffer = token_to_kv_pool.get_value_buffer(layer_id)

        src_k = k_buffer.index_select(0, source_slots)
        src_v = v_buffer.index_select(0, source_slots)

        if rotary_dim < src_k.shape[-1]:
            rot_k = src_k[..., :rotary_dim]
            pass_k = src_k[..., rotary_dim:]
            rot_k = reposition_key(
                rot_k,
                delta_positions=delta_positions,
                inv_freq=inv_freq,
                is_neox_style=is_neox_style,
            )
            new_k = torch.cat((rot_k, pass_k), dim=-1)
        else:
            new_k = reposition_key(
                src_k,
                delta_positions=delta_positions,
                inv_freq=inv_freq,
                is_neox_style=is_neox_style,
            )

        k_buffer.index_copy_(0, dest_slots, new_k.to(k_buffer.dtype))
        v_buffer.index_copy_(0, dest_slots, src_v)


def copy_kv_to_new_slots(
    *,
    token_to_kv_pool: "KVCache",
    layer_ids: Sequence[int],
    source_slots: torch.Tensor,
    dest_slots: torch.Tensor,
) -> None:
    """Straight per-layer K/V copy, no repositioning -- used to register a
    finished request's span into the sub-context pool at its *original*
    position. The stored K stays RoPE'd at that original absolute position;
    ``reposition_key`` corrects it for wherever it's reused later
    (``materialize_reused_kv``). Correct regardless of RoPE style since
    nothing here depends on position."""
    if source_slots.numel() == 0:
        return
    for layer_id in layer_ids:
        k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
        v_buffer = token_to_kv_pool.get_value_buffer(layer_id)
        k_buffer.index_copy_(0, dest_slots, k_buffer.index_select(0, source_slots))
        v_buffer.index_copy_(0, dest_slots, v_buffer.index_select(0, source_slots))


def attention_backend_supports_subcontext_reuse(backend_str: str) -> bool:
    """Whether ``backend_str`` reads the KV context for a query token from
    the full per-request page table (``req_to_token_pool``, sized off
    ``seq_lens``) rather than splitting it into a "ragged" self-attention
    pass over just this step's own query tokens plus a separately-sized
    "paged" pass over the literal prefix.

    That split (FlashInfer's ``use_ragged`` fast path, on by default:
    ``flashinfer_backend.py``'s ``update_single_wrapper``/``update`` size
    the paged half off ``extend_prefix_lens`` alone) assumes a request's
    forward-pass tokens are exactly its non-cached suffix -- true for
    ordinary prefix caching, false here, where the "new" tokens can be a
    non-contiguous subset with a subcontext-reused span sitting in a gap
    the ragged pass never reads. FlashAttention's extend path has no such
    split: ``flashattention_backend.py`` always sizes ``cu_seqlens_k`` /
    ``page_table`` off the full ``seq_lens``, using ``extend_seq_lens``
    only for the query side, so it's unconditionally safe. FlashInfer is
    only safe with its ragged path forced off
    (``SGLANG_FLASHINFER_USE_PAGED=1``).
    """
    if backend_str == "fa3":
        return True
    if backend_str == "flashinfer":
        return envs.SGLANG_FLASHINFER_USE_PAGED.get()
    return False


def materialize_reused_kv_for_batch(
    *,
    model: torch.nn.Module,
    token_to_kv_pool: "KVCache",
    plan: "SubcontextMaterializePlan",
) -> None:
    """Convenience wrapper for ``managers/tp_worker.py``: resolves the
    model's rotary embedding and the KV pool's layer range, then
    materializes every layer's reused K/V for one batch's plan. Takes the
    model and pool directly rather than a whole ``ModelRunner`` -- they're
    the only two things this needs (see
    ``.claude/rules/general-code-style.md``: "pass what you need, not the
    god object").

    Raises rather than silently skipping if the model has no single shared
    RotaryEmbedding (mixed RoPE bases, or a position-independent attention
    model): a resolved plan with nowhere to apply it means the batch's
    reused tokens were already excluded from the forward pass, so skipping
    materialization would leave their KV slots stale rather than degrading
    gracefully. This should have been rejected at server startup instead
    of reached here -- see the MVP scope note in
    ``docs/docs/advanced_features/subcontext_kv_cache.mdx``.
    """
    rotary_emb = find_rotary_embedding(model)
    if rotary_emb is None:
        raise RuntimeError(
            "Sub-context KV reuse matched a request but the model has no "
            "single shared RotaryEmbedding module; unsupported by the "
            "current MVP scope (dense, standard-RoPE models only)."
        )
    inv_freq = rotary_emb._compute_inv_freq(rotary_emb.base).to(
        device=plan.source_slots.device, dtype=torch.float32
    )
    materialize_reused_kv(
        token_to_kv_pool=token_to_kv_pool,
        layer_ids=range(token_to_kv_pool.start_layer, token_to_kv_pool.end_layer + 1),
        source_slots=plan.source_slots,
        dest_slots=plan.dest_slots,
        delta_positions=plan.delta_positions,
        inv_freq=inv_freq,
        is_neox_style=rotary_emb.is_neox_style,
        rotary_dim=rotary_emb.rotary_dim,
    )
