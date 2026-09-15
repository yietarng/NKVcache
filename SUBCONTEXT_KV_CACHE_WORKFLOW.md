# Sub-Context KV Cache (NOC) — Workflow

Status: the full pipeline — matching, reuse, and registration — is
implemented and unit-tested (27 cases,
`test/registered/unit/mem_cache/test_subcontext_cache.py`). Nothing here
has been exercised against a real GPU/model forward pass; see
[What still needs GPU validation](#what-still-needs-gpu-validation) before
treating this as production-ready.

## Request lifecycle

```
 client              tokenizer_manager      scheduler.py /           schedule_batch.py         tp_worker.py /
 /generate             (io_struct.py)      batch_result_processor.py                       forward_batch_info.py
   |                        |                      |                        |                        |
   | subcontext_tags        |                      |                        |                        |
   |----------------------->|                      |                        |                        |
   |                        | TokenizedGenerate     |                        |                        |
   |                        | ReqInput.subcontext_  |                        |                        |
   |                        | tags                  |                        |                        |
   |                        |--------------------->|                        |                        |
   |                        |                      | Req(subcontext_tags=..)|                        |
   |                        |                      |----------------------->|                        |
   |                        |                      |                        | init_next_round_input:  |
   |                        |                      |                        |  1. ordinary prefix     |
   |                        |                      |                        |     match (unchanged)   |
   |                        |                      |                        |  2. subcontext_scan()   |
   |                        |                      |                        |     over the unmatched  |
   |                        |                      |                        |     suffix, end capped  |
   |                        |                      |                        |     at input_len - 1    |
   |                        |                      |                        |  -> req.subcontext_     |
   |                        |                      |                        |     matches / plans     |
   |                        |                      |                        |                        |
   |                        |                      |                        | prepare_for_extend:     |
   |                        |                      |                        |  - eligibility gate     |
   |                        |                      |                        |  - alloc KV (full range)|
   |                        |                      |                        |  - plan_request_extend  |
   |                        |                      |                        |    + build_batch_       |
   |                        |                      |                        |    subcontext_plan      |
   |                        |                      |                        |  -> gathered input_ids, |
   |                        |                      |                        |     shrunk out_cache_loc|
   |                        |                      |                        |     /extend_lens,       |
   |                        |                      |                        |     subcontext_         |
   |                        |                      |                        |     positions +         |
   |                        |                      |                        |     materialize_plan    |
   |                        |                      |                        |                        |
   |                        |                      |                        |                        | ForwardBatch.init_new:
   |                        |                      |                        |                        |  positions overridden
   |                        |                      |                        |                        |  from subcontext_
   |                        |                      |                        |                        |  positions
   |                        |                      |                        |                        |
   |                        |                      |                        |                        | materialize_reused_kv_
   |                        |                      |                        |                        | for_batch: copy+rotate
   |                        |                      |                        |                        | K, copy V, per layer,
   |                        |                      |                        |                        | into new slots
   |                        |                      |                        |                        |
   |                        |                      |                        |                        | model_runner.forward():
   |                        |                      |                        |                        | only surviving tokens
   |                        |                      |                        |                        | run through the model;
   |                        |                      |                        |                        | attention reads the
   |                        |                      |                        |                        | FULL context (prefix +
   |                        |                      |                        |                        | materialized + fresh)
   |                        |                      |                        |                        |
   |                        |                      | req.finished():        |                        |
   |                        |                      |  register_subcontext_  |                        |
   |                        |                      |  entries() -- copies   |                        |
   |                        |                      |  tagged spans' K/V     |                        |
   |                        |                      |  into the sub-context  |                        |
   |                        |                      |  pool, registers them  |                        |
   |                        |                      |  BEFORE release_kv_    |                        |
   |                        |                      |  cache frees the row   |                        |
   |<----------------------------------------------|                                                 |
   | generated tokens                                                    (a LATER request's scan in
   |                                                                       step 2 can now find and
   |                                                                       reuse what this one
   |                                                                       registered)
```

## Step-by-step, with the exact function/file at each step

### 0. Startup (once, per server process)

`Scheduler.maybe_init_subcontext_index()` — `python/sglang/srt/managers/scheduler.py`

- If `--enable-subcontext-kv-cache` is off: `self.subcontext_index = None`. Every
  later step below checks for this and no-ops, so the server behaves exactly
  as it did before this feature existed.
- If on: validates the loaded model (`kv_materialize.find_rotary_embedding` —
  must have exactly one shared `RotaryEmbedding`, i.e. dense/standard-RoPE)
  and the attention backend (`kv_materialize.attention_backend_supports_
  subcontext_reuse` — must be `fa3`, or `flashinfer` with
  `SGLANG_FLASHINFER_USE_PAGED=1`; see
  [Attention-backend constraint](#attention-backend-constraint) below).
  Raises at startup if either check fails.
- Reserves `--subcontext-kv-cache-tokens` (default 4096) physical KV-pool
  slots once, via `token_to_kv_pool_allocator.alloc(...)` — a permanent
  reservation, never returned to the ordinary allocator. Raises at startup
  if the pool doesn't have that much free capacity.
- Constructs `self.subcontext_index = SubContextIndex()`
  (`mem_cache/subcontext/subcontext_index.py`) and binds it to those
  reserved slots (`bind_slots`) — a ring buffer, currently empty.

### 1. Request enters the system

- Client sends `/generate` with an optional `subcontext_tags` field:
  `[{"subcontext_id": "...", "start": <token offset>, "end": <token offset>}]`
  (`managers/io_struct.py: GenerateReqInput.subcontext_tags`).
- `TokenizerManager` copies it onto `TokenizedGenerateReqInput.subcontext_tags`
  (`managers/tokenizer_manager.py`).
- `Scheduler.handle_generate_request` builds a `Req` from it, passing
  `subcontext_tags=recv_req.subcontext_tags` (`managers/scheduler.py`), stored
  as `req.subcontext_tags` (`managers/schedule_batch.py: Req.__init__`).

### 2. Matching, once per scheduling round

`Req.init_next_round_input(tree_cache, subcontext_index=self.subcontext_index, subcontext_brz_window=...)`
— `managers/schedule_batch.py`, called from `managers/scheduler.py`'s main
prefill admission loop.

1. The ordinary RadixCache prefix match runs first, completely unchanged
   (`tree_cache.match_prefix(...)`) — this is still the primary, cheapest
   reuse path.
2. If `subcontext_index` is set and there are tokens left unmatched: resolve
   `req.subcontext_tags` into `SubContextTag` objects, then call
   `subcontext_scan(index, token_ids, explicit_tags=tags, start=len(prefix_indices), end=input_len-1)`
   (`mem_cache/subcontext/subcontext_scanner.py`). This checks explicit tags
   directly against the index, then greedily scans the rest for the longest
   verified match at each position, using the index's probe-hash buckets as
   a cheap candidate filter (`mem_cache/subcontext/subcontext_index.py`).
   `end=input_len-1` guarantees at least one token is always left over to
   compute a logit and sample from.
3. The full, query-ordered match list becomes a list of `RecomputePlan`s via
   `plan_boundary_recompute_zones(matches, k=subcontext_brz_window)`
   (`mem_cache/subcontext/deviation_recompute.py`): every match's own
   leading `k` tokens are recomputed (CacheBlend/EPIC-style correction,
   `First_k(B)`); a match's trailing `k` tokens are *additionally*
   recomputed when the next match starts exactly where it ends -- a
   genuine stitched boundary, no glue between them (`Last_k(A)`). `k=0`
   is pure reuse, no correction.
4. Result: `req.subcontext_matches` and `req.subcontext_recompute_plans`.

### 3. Batch construction

`ScheduleBatch.prepare_for_extend()` — `managers/schedule_batch.py`.

1. **Eligibility gate.** Batch-level: `return_logprob` off, not DLLM, not
   using the mamba extra buffer. Per-request: has a non-empty
   `subcontext_recompute_plans`, no multimodal inputs, no positional-embed
   overrides, no `input_embeds`. Anything that fails the gate takes the
   exact original code path (full contiguous `input_ids` slice) — the new
   logic only ever touches eligible requests.
2. **KV allocation happens first, unaffected**, sized off the *full* logical
   extend range (`alloc_for_extend`) — every position gets a physical slot
   whether it ends up reused or freshly computed.
3. For each eligible request: `plan_request_extend()`
   (`mem_cache/subcontext/extend_plan.py`) splits its extend range into
   `surviving_local` offsets (need a real forward pass) and `reused_local`
   offsets (get materialized instead), dropping any match that isn't
   *entirely* inside this round's extend range (so chunked prefill never
   partially materializes a match).
4. `build_batch_subcontext_plan()` (`extend_plan.py`) applies that split to
   the batch's already-allocated `out_cache_loc`: narrows it to just the
   surviving slots, rebinds `self.extend_lens` / `self.extend_num_tokens` to
   the surviving counts, builds `self.subcontext_positions` (true logical
   positions for the surviving tokens) and `self.subcontext_materialize_plan`
   (source slot / dest slot / delta-position triples for every reused
   token). `input_ids` becomes a *gather* over the surviving positions
   instead of a contiguous slice.

### 4. Forward pass

`TpModelWorker.forward_batch_generation()` — `managers/tp_worker.py`.

1. `ForwardBatch.init_new(batch, model_runner, ...)` builds the forward
   batch; inside it, `model_executor/forward_batch_info.py` overrides
   `ret.positions` from `batch.subcontext_positions` when set (in the same
   branch that already handles DLLM/spec-info position overrides).
2. If `batch.subcontext_materialize_plan` is set:
   `materialize_reused_kv_for_batch(model=..., token_to_kv_pool=..., plan=...)`
   (`mem_cache/subcontext/kv_materialize.py`) runs **before** the model
   forward: for every transformer layer, gather the source chunk's K/V at
   its stored slots, reposition K by the offset delta
   (`layers/rotary_embedding/reposition.py`, using RoPE's additive rotation
   composition), copy V unchanged, and write both into the new destination
   slots. Pure data movement, no model computation.
3. `model_runner.forward(forward_batch)` runs. Only the surviving tokens are
   actually fed through the model (embeddings, attention, MLP, every
   layer) — the reused tokens' KV is already sitting in the pool from step
   2. Attention for the surviving tokens reads the *full* logical context
   (literal prefix + materialized reused spans + their own freshly computed
   K/V) via the normal page-table read (`req_to_token_pool`) — see
   [Attention-backend constraint](#attention-backend-constraint) for why
   this needs a specific attention backend.

### 5. Registration — populating the index for later requests

`mem_cache/common.py: register_subcontext_entries(req, subcontext_index, req_to_token_pool, token_to_kv_pool)`
— called from `managers/scheduler_components/batch_result_processor.py`, at
the primary generation-finished site (`if req.finished(): ...`), **before**
`release_kv_cache(...)` (which is what frees or reassigns
`req.kv.req_pool_idx`'s row — registration has to read it first).

1. Only registers `req.subcontext_tags` (explicitly-tagged spans) —
   automatic discovery of arbitrary recurring content is a separate,
   unimplemented policy question (see the design doc's Registration
   section for why).
2. Only requests that reach `finished()` are registered, and only tags
   with `end <= req.effective_kv_committed_len()` (the same bound
   `release_kv_cache` itself uses) — never a still-running request (its
   span could still be retracted) or a span reaching into uncommitted/
   reclaimed territory.
3. For each valid tag: read its physical slots off
   `req_to_token_pool.req_to_token[req.kv.req_pool_idx]`, call
   `subcontext_index.register(token_ids=..., orig_position=start,
   subcontext_id=...)` to reserve space in the sub-context pool's ring
   buffer (a dedicated, permanently-reserved slice of the KV pool — see
   below), then `kv_materialize.copy_kv_to_new_slots` to actually copy the
   K/V there (no repositioning — stored at its original position, corrected
   later at consumption time by step 2's `reposition_key`).
4. `register()` returns `None` (nothing copied) when the content is
   already registered, too large for the whole pool, or no unreferenced
   ring space could be found right now — never by evicting an entry a
   live request is still consuming (`inc_ref`/`dec_ref`-protected).

**Why a dedicated pool, not an alias into the radix cache's own slots?**
Aliasing would need `SubContextIndex` to be notified whenever the radix
tree's own eviction reclaims an overlapping node — no such hook exists
today. A dedicated ring buffer (`--subcontext-kv-cache-tokens`, reserved
once at startup, step 0) avoids that coordination problem entirely, at the
cost of a fixed, explicit memory reservation.

## Attention-backend constraint

FlashInfer's default prefill path splits attention into a *ragged*
self-attention pass over just this step's own query tokens, plus a
*paged* pass sized off `extend_prefix_lens` alone — never `seq_lens`. A
subcontext-reused span sitting in a gap between two surviving positions
would fall into neither pass and silently never get attended to.
FlashAttention's extend path has no such split — it always sizes its KV
window off the full `seq_lens`. This is why step 0 gates the backend:
`fa3` is allowlisted unconditionally; `flashinfer` only with
`SGLANG_FLASHINFER_USE_PAGED=1` forcing its ragged path off. See the
design doc (`docs/docs/advanced_features/subcontext_kv_cache.mdx`, inside
the vendored tree) for the full writeup.

## File reference

### `mem_cache/subcontext/`

| File | Role |
|---|---|
| `subcontext_types.py` | `SubContextTag`, `SubContextEntry`, `SubContextMatch`, `RecomputePlan`, `SubcontextMaterializePlan` |
| `subcontext_index.py` | `SubContextIndex` — content-hash lookup + ring-buffer physical slot reservation/eviction |
| `subcontext_scanner.py` | `scan()` — explicit-tag + automatic longest-match detection over the unmatched suffix, capped short of the request's last token |
| `deviation_recompute.py` | `plan_boundary_recompute_zones` — Boundary Recompute Zone policy |
| `extend_plan.py` | `plan_request_extend`, `build_batch_subcontext_plan` — per-request and per-batch split of the extend range |
| `kv_materialize.py` | `materialize_reused_kv`, `materialize_reused_kv_for_batch`, `copy_kv_to_new_slots`, `find_rotary_embedding`, `attention_backend_supports_subcontext_reuse` |

### Other files

| File | Role |
|---|---|
| `layers/rotary_embedding/reposition.py` | RoPE delta-rotation math (`reposition_key`) |
| `mem_cache/common.py` | `register_subcontext_entries` (new), `release_kv_cache` (existing, now called after it) |
| `test/registered/unit/mem_cache/test_subcontext_cache.py` | 27 unit tests |
| `docs/docs/advanced_features/subcontext_kv_cache.mdx` | Full design doc |

### Modified files

| File | What changed |
|---|---|
| `managers/schedule_batch.py` | `Req.subcontext_tags` / `subcontext_matches` / `subcontext_recompute_plans`; the scan call (with `end` cap) in `init_next_round_input`; the eligibility gate, gather, and `out_cache_loc`/`extend_lens` rewrite in `prepare_for_extend` |
| `managers/scheduler.py` | `maybe_init_subcontext_index` (model + backend validation, pool reservation, index construction); passes `subcontext_tags` into `Req(...)`; passes `self.subcontext_index` into the main prefill admission's `init_next_round_input` call and into `SchedulerBatchResultProcessor` |
| `managers/scheduler_components/batch_result_processor.py` | `subcontext_index` field; calls `register_subcontext_entries` at the generation-finished site, before `release_kv_cache` |
| `managers/tp_worker.py` | Calls `materialize_reused_kv_for_batch` right after `ForwardBatch.init_new`, before the model forward |
| `model_executor/forward_batch_info.py` | `ForwardBatch.init_new`'s `positions` override branch |
| `managers/io_struct.py` | `GenerateReqInput.subcontext_tags`, `TokenizedGenerateReqInput.subcontext_tags` |
| `managers/tokenizer_manager.py` | Propagates `subcontext_tags` into `TokenizedGenerateReqInput` |
| `arg_groups/fields/memory.py` | `--enable-subcontext-kv-cache`, `--subcontext-brz-window`, `--subcontext-kv-cache-tokens` CLI flags |

All paths above are relative to `sglang/python/sglang/srt/` (or `sglang/test/`,
`sglang/docs/`) inside this repo, on branch
`claude/sglang-subcontext-kv-cache-dtv6r3`.

## What still needs GPU validation

Every module is either pure-CPU unit tested (index ring buffer, scanner,
rotation math, materializer against a fake pool, the batch-plan transform)
or import/syntax verified end-to-end (the real dependency chain, resolved
in a CPU-only sandbox, imports every touched module cleanly). None of it
has run against a real GPU, model, or attention kernel. In particular:

- The core mechanism's central claim — that an attention backend reading
  the full page-table row produces correct results for a non-contiguous
  set of "new" query tokens — is architecturally verified by reading the
  backend source (`flashattention_backend.py`), not by running it.
- `register_subcontext_entries`'s physical-slot bookkeeping (reading a
  live `req_to_token_pool` row, writing into the reserved pool) has no
  unit test, since exercising it meaningfully needs a real pool.

Recommended smoke test before any real traffic: a small Llama-style model,
`--attention-backend fa3 --enable-subcontext-kv-cache`, a first request
tagging a span (`subcontext_tags`), a second request whose prompt repeats
that exact span at a different offset, `--subcontext-brz-window 0`
(pure reuse, easiest to reason about) — confirm the second request's
output matches a baseline run with the feature off.
