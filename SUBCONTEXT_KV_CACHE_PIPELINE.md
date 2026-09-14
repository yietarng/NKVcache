# Sub-Context KV Cache (NOC) — Prefill & Generation Pipeline

Focused reference for how prefill and decode work with
`--enable-subcontext-kv-cache` on. For the full request lifecycle across
every file, see `SUBCONTEXT_KV_CACHE_WORKFLOW.md`; for the architecture
rationale, see `sglang/docs/docs/advanced_features/subcontext_kv_cache.mdx`
inside the vendored tree.

Status: implemented and unit-tested (27 cases), not yet run against a real
GPU/model. Branch `claude/sglang-subcontext-kv-cache-dtv6r3`.

## Prefill pipeline

Sub-context KV cache only touches prefill. One scheduling round, one
request in `ForwardMode.EXTEND`:

```
1. Tokenize + admit
   Req built with req.subcontext_tags (from the client's /generate call)

2. Req.init_next_round_input()                         [schedule_batch.py]
   a. tree_cache.match_prefix()  — ordinary RadixCache prefix match, unchanged
   b. subcontext_scan(index, tokens, start=prefix_len, end=input_len-1)
      → req.subcontext_matches / req.subcontext_recompute_plans
      (only runs if --enable-subcontext-kv-cache; empty otherwise)

3. ScheduleBatch.prepare_for_extend()                   [schedule_batch.py]
   a. eligibility gate (no multimodal / embeds / logprobs / DLLM / mamba)
   b. alloc_for_extend()  — KV slots for the FULL extend range, unaffected
   c. plan_request_extend() + build_batch_subcontext_plan()
      → splits the range into "surviving" (needs forward) vs "reused"
      → gathers input_ids to just the surviving tokens
      → shrinks out_cache_loc / extend_lens to match
      → builds subcontext_positions + subcontext_materialize_plan

4. TpModelWorker.forward_batch_generation()              [tp_worker.py]
   a. ForwardBatch.init_new()  — positions overridden from subcontext_positions
   b. materialize_reused_kv_for_batch()  — BEFORE the model runs:
      per layer, copy the reused chunk's K/V into the new slots,
      reposition K by the offset delta (RoPE rotation)
   c. model_runner.forward()  — only surviving tokens actually go through
      embeddings → attention → MLP, every layer. Their attention reads the
      FULL context (prefix + materialized reused spans + their own fresh
      K/V) via the normal page-table read — nothing else changes.

5. req.finished()? (usually not yet — prefill just produced token 1)
   If it did finish here (e.g. max_new_tokens=1): register_subcontext_entries()
   runs now, before release_kv_cache. Otherwise this waits for step 8.
```

Steps 3c and 4b are skipped entirely — same code path as before this
feature existed — for any request that doesn't hit the eligibility gate or
has no matches. That's the majority case until something has actually been
registered.

## Generation (decode) pipeline

**Sub-context KV cache does not touch decode at all.** Once prefill has
produced the first token, decode is the standard one-token-at-a-time loop,
completely unmodified:

```
repeat until finished:
    ForwardBatch.init_new()  — forward_mode = DECODE, ordinary positions
                                (subcontext_positions is only set for EXTEND;
                                 the override branch never fires here)
    out_cache_loc = alloc_for_decode()  — one new slot per request, ordinary
    model_runner.forward()  — one token, attends over the full committed
                               context (which already includes whatever
                               was materialized during prefill)
    sample next token, append to output_ids
```

There's no gather, no materialization, no eligibility gate here — decode
reads whatever prefill already wrote into the KV pool (real or
materialized) via the same `req_to_token_pool` row it's always used. This
is intentional: sub-context reuse only pays off for prefill's full-context
recompute; decode is already one token, nothing to reuse there.

## Request finish → registration

```
8. req.finished()                              [batch_result_processor.py]
   a. register_subcontext_entries()  — BEFORE release_kv_cache:
      for each tag in req.subcontext_tags with end <= effective_kv_committed_len:
        read the span's physical slots off req_to_token_pool's row
        subcontext_index.register(...) — reserve ring-buffer space
        copy_kv_to_new_slots() — copy K/V into the sub-context pool
   b. release_kv_cache()  — frees/reassigns the row, inserts into RadixCache
                             as normal (unaffected by any of the above)
```

This is what a **later** request's step 2 scan finds. So the cross-request
timeline for two requests sharing a tagged span looks like:

```
Request A:  prefill (no match yet) → decode → finish → registers the span
Request B:  prefill → subcontext_scan finds A's registered span
                     → prepare_for_extend gathers around it
                     → materialize_reused_kv_for_batch reuses A's KV
                     → decode (unmodified) → finish
```
