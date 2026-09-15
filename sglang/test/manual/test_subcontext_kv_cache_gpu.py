"""
Sub-context KV Cache (NOC) — Real-GPU Correctness + Trace Smoke Test
=====================================================================

Everything under test/registered/unit/mem_cache/test_subcontext_cache.py runs
on CPU against fake tensors and a fake pool: it pins down the scanning,
planning, and BRZ math, but never a real model, real CUDA kernels, or a real
attention backend. This script is the GPU-side complement, meant to be run
by hand on real hardware -- it is NOT auto-discovered by run_suite.py (see
test/manual/README convention), because CI in this sandbox has no GPU to
verify it against.

What it checks
---------------
1. Correctness: launches a real small model 3x (feature off / BRZ=0 / BRZ=32)
   against the same prompt pair -- a "shared block" (e.g. a tool output)
   registered by an early request, then recurring at a different offset in a
   later request -- and compares greedy completions + per-token logprobs.
   Feature-off is ground truth. BRZ=0 (pure reposition, no repair) is
   expected to be the noisiest; BRZ=32 (the shipped default) should track
   ground truth much more closely. This is a smoke comparison, not a
   statistical accuracy bar -- read the printed diffs yourself.

2. Trace capture: uses SGLang's built-in profiler endpoints (/start_profile,
   /stop_profile) to capture a real torch profiler trace for the request
   that hits the reuse path, so you can open it in chrome://tracing (or
   any perfetto-compatible viewer) and confirm directly:
     - the extend/prefill step's token count is smaller than the full
       shared-block length (proof the reused span was NOT recomputed from
       scratch),
     - the reposition (RoPE delta) and kv_materialize copy kernels actually
       appear on the timeline for that step.
   Token-level output matching alone can't distinguish "reuse worked" from
   "reuse was silently skipped and everything was recomputed" if the model
   is deterministic enough that both paths agree -- the trace is the only
   way to see which code path actually ran.

Usage
-----
    python test/manual/test_subcontext_kv_cache_gpu.py \\
        --model-path <a small dense, standard-RoPE HF model> \\
        --trace-dir /tmp/subcontext_traces

Requires a real CUDA GPU and the fa3 attention backend (or flashinfer with
SGLANG_FLASHINFER_USE_PAGED=1 -- see attention_backend_supports_subcontext_reuse
in python/sglang/srt/managers/scheduler.py). MLA / mrope / SWA / Mamba /
CUDA-graph models are out of MVP scope and will fail fast at server startup.
"""

import argparse
import time

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

# A long-ish filler so the shared block is unambiguously a multi-token span,
# not something a short prefix match could explain away.
_SHARED_BLOCK = (
    "Tool output: the quarterly report shows revenue of $4.2M, up 12% "
    "year-over-year, driven primarily by the enterprise segment which grew "
    "18% while the SMB segment grew only 4%. Operating margin held steady "
    "at 22%. Full breakdown attached."
)

_PROMPT_A = (
    f"You are a financial analyst assistant. {_SHARED_BLOCK} "
    "Summarize the enterprise segment performance in one sentence."
)
_PROMPT_B = (
    "You are a financial analyst assistant. Here is some unrelated context "
    "about last month's server incident, included only to shift the shared "
    f"block to a different token offset than before. {_SHARED_BLOCK} "
    "Summarize the SMB segment performance in one sentence."
)


def _shared_block_token_span(tokenizer, prompt: str, block: str) -> tuple:
    """Approximate token offsets of `block` within `prompt`, via encode-length
    deltas. Good enough for a manual smoke test; BPE merges right at the
    prefix/block seam can shift this by a token or two, which only widens
    or narrows the registered span slightly and does not invalidate the
    comparison -- the scanner re-verifies content on every match regardless.
    """
    prefix = prompt[: prompt.index(block)]
    start = len(tokenizer.encode(prefix, add_special_tokens=True))
    end = len(tokenizer.encode(prefix + block, add_special_tokens=True))
    return start, end


def _generate(base_url: str, prompt: str, subcontext_tags=None):
    payload = {
        "text": prompt,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 32},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if subcontext_tags is not None:
        payload["subcontext_tags"] = subcontext_tags
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _run_one_config(
    *,
    model_path: str,
    tokenizer,
    label: str,
    extra_args: list,
    trace_dir: str,
    feature_enabled: bool,
):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    base_url = DEFAULT_URL_FOR_TEST
    process = popen_launch_server(
        model_path,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=extra_args,
    )
    try:
        # Registration only happens for explicitly-tagged spans (see
        # register_subcontext_entries in mem_cache/common.py); the
        # automatic content-hash scanner only *matches* on later requests,
        # it never registers. Request A must tag the block for anything to
        # be reusable when request B's automatic scan finds it again.
        tags = None
        if feature_enabled:
            start, end = _shared_block_token_span(
                tokenizer, _PROMPT_A, _SHARED_BLOCK
            )
            tags = [
                {"subcontext_id": "financial_summary_block", "start": start, "end": end}
            ]
        out_a = _generate(base_url, _PROMPT_A, subcontext_tags=tags)
        print("Prompt A completion:", out_a["text"])

        if feature_enabled:
            requests.post(
                f"{base_url}/start_profile",
                json={"output_dir": trace_dir, "activities": ["CPU", "GPU"]},
                timeout=30,
            )

        out_b = _generate(base_url, _PROMPT_B)
        print("Prompt B completion:", out_b["text"])

        if feature_enabled:
            requests.post(f"{base_url}/stop_profile", timeout=30)
            time.sleep(2)  # profiler flush is async
            print(f"Trace written under: {trace_dir}")

        logprobs_b = [
            lp[0] for lp in out_b["meta_info"]["output_token_logprobs"]
        ]
        return {
            "text_a": out_a["text"],
            "text_b": out_b["text"],
            "logprobs_b": logprobs_b,
        }
    finally:
        kill_process_tree(process.pid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-dir", default="/tmp/subcontext_traces")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    configs = [
        ("baseline (feature off)", []),
        (
            "subcontext reuse, BRZ=0 (pure reposition, no repair)",
            [
                "--enable-subcontext-kv-cache",
                "--subcontext-brz-window",
                "0",
                "--attention-backend",
                "fa3",
            ],
        ),
        (
            "subcontext reuse, BRZ=32 (shipped default)",
            [
                "--enable-subcontext-kv-cache",
                "--subcontext-brz-window",
                "32",
                "--attention-backend",
                "fa3",
            ],
        ),
    ]

    results = {}
    for label, extra_args in configs:
        feature_enabled = "--enable-subcontext-kv-cache" in extra_args
        results[label] = _run_one_config(
            model_path=args.model_path,
            tokenizer=tokenizer,
            label=label,
            extra_args=extra_args,
            trace_dir=args.trace_dir,
            feature_enabled=feature_enabled,
        )

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    baseline = results["baseline (feature off)"]
    for label, result in results.items():
        print(f"\n[{label}]")
        print(f"  completion B: {result['text_b']!r}")
        if label != "baseline (feature off)":
            match = result["text_b"] == baseline["text_b"]
            print(f"  text matches baseline: {match}")
            n = min(len(result["logprobs_b"]), len(baseline["logprobs_b"]))
            diffs = [
                abs(result["logprobs_b"][i] - baseline["logprobs_b"][i])
                for i in range(n)
            ]
            if diffs:
                print(
                    f"  logprob abs diff vs baseline: "
                    f"max={max(diffs):.4f} mean={sum(diffs) / len(diffs):.4f}"
                )

    print(
        "\nNow open the BRZ=0 and BRZ=32 traces from "
        f"{args.trace_dir} in chrome://tracing and confirm: (a) the extend "
        "step for prompt B's forward pass processes fewer tokens than the "
        "full prompt length, and (b) reposition/kv_materialize ops appear "
        "on the timeline."
    )


if __name__ == "__main__":
    main()
