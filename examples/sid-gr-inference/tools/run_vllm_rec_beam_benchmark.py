# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the vLLM REC_BEAM path on a shared token-id JSONL workload."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tool_utils import numeric_median, read_jsonl, write_json


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import BeamSearchParams
    from vllm.v1.engine.async_llm import AsyncLLM

    workload = _load_workload(
        Path(args.workload_jsonl),
        limit=args.requests,
        context_len=args.context_len,
    )
    engine_args = AsyncEngineArgs(
        model=args.model_dir,
        dtype="bfloat16",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False,
        disable_log_stats=True,
        additional_config={"use_rec_beam_search": True},
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    try:
        for warmup_index in range(args.warmup_runs):
            await _run_once(
                engine,
                workload,
                args,
                run_prefix=f"warmup-{warmup_index}",
                record_outputs=False,
            )
        measured = [
            await _run_once(
                engine,
                workload,
                args,
                run_prefix=f"measured-{run_index}",
                record_outputs=True,
            )
            for run_index in range(args.repeat)
        ]
    finally:
        engine.shutdown()

    return {
        "schema_version": "three_way_offline_v1",
        "framework": "vllm",
        "model_dir": args.model_dir,
        "workload_jsonl": args.workload_jsonl,
        "context_len": args.context_len,
        "decode_steps": args.decode_steps,
        "beam_width": args.beam_width,
        "requests": len(workload),
        "warmup_runs": args.warmup_runs,
        "repeat": args.repeat,
        "prefix_cache_enabled": False,
        "sampling_params": {
            "beam_width": args.beam_width,
            "max_tokens": args.decode_steps,
            "temperature": 0.0,
            "ignore_eos": True,
            "length_penalty": 1.0,
            "detokenize": False,
        },
        "engine_args": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        },
        **_metric_summaries(measured),
        "runs": measured,
        "environment": _environment(torch),
        "vllm_commit": _git_commit(args.vllm_repo),
    }


async def _run_once(
    engine: Any,
    workload: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    run_prefix: str,
    record_outputs: bool,
) -> dict[str, Any]:
    from vllm.sampling_params import BeamSearchParams

    gate = asyncio.Event()
    params = BeamSearchParams(
        beam_width=args.beam_width,
        max_tokens=args.decode_steps,
        temperature=0.0,
        ignore_eos=True,
        length_penalty=1.0,
        detokenize=False,
    )

    async def run_request(index: int, row: dict[str, Any]) -> dict[str, Any]:
        await gate.wait()
        submitted_at = time.perf_counter()
        request_output = None
        async for output in engine.rec_beam_search(
            prompt={
                "type": "token",
                "prompt_token_ids": row["input_ids"],
            },
            params=params,
            request_id=f"{run_prefix}-{row['request_id']}",
        ):
            request_output = output
        completed_at = time.perf_counter()
        if request_output is None:
            raise RuntimeError(f"vLLM returned no output for {row['request_id']}")
        return {
            "index": index,
            "row": row,
            "latency_ms": (completed_at - submitted_at) * 1000.0,
            "output": request_output,
        }

    tasks = [
        asyncio.create_task(run_request(index, row))
        for index, row in enumerate(workload)
    ]
    started_at = time.perf_counter()
    gate.set()
    request_results = await asyncio.gather(*tasks)
    wall_ms = (time.perf_counter() - started_at) * 1000.0
    request_results.sort(key=lambda item: item["index"])

    normalize_started_at = time.perf_counter()
    outputs = (
        [
            _normalize_output(result["row"], result["output"], args)
            for result in request_results
        ]
        if record_outputs
        else []
    )
    output_normalization_ms = (time.perf_counter() - normalize_started_at) * 1000.0
    latencies = [float(result["latency_ms"]) for result in request_results]
    elapsed_s = wall_ms / 1000.0
    request_count = len(workload)
    generated_tokens = request_count * args.decode_steps
    beam_candidates = generated_tokens * args.beam_width
    return {
        "wall_ms": wall_ms,
        "qps": request_count / elapsed_s if elapsed_s else None,
        "generated_tokens_per_s": generated_tokens / elapsed_s if elapsed_s else None,
        "beam_candidates_per_s": beam_candidates / elapsed_s if elapsed_s else None,
        "request_latencies_ms": latencies,
        "request_latency_ms_p50": _percentile(latencies, 0.50),
        "request_latency_ms_p95": _percentile(latencies, 0.95),
        "output_normalization_ms": output_normalization_ms,
        "outputs": outputs,
    }


def _normalize_output(
    row: dict[str, Any], output: Any, args: argparse.Namespace
) -> dict[str, Any]:
    beams = [
        {
            "rank": int(candidate.index),
            "token_ids": [int(token_id) for token_id in candidate.token_ids],
            "score": (
                float(candidate.cumulative_logprob)
                if candidate.cumulative_logprob is not None
                else None
            ),
        }
        for candidate in output.outputs
    ]
    beams.sort(key=lambda candidate: candidate["rank"])
    if len(beams) != args.beam_width:
        raise RuntimeError(
            f"vLLM returned {len(beams)} candidates for {row['request_id']}; "
            f"expected {args.beam_width}"
        )
    if any(beam["score"] is None for beam in beams):
        raise RuntimeError(
            f"vLLM returned candidates without cumulative logprob for "
            f"{row['request_id']}"
        )
    wrong_lengths = [
        beam["rank"]
        for beam in beams
        if len(beam["token_ids"]) != args.decode_steps
    ]
    if wrong_lengths:
        raise RuntimeError(
            f"vLLM returned non-{args.decode_steps}-token candidates for "
            f"{row['request_id']}: ranks={wrong_lengths[:8]}"
        )
    return {
        "request_id": row["request_id"],
        "workload_id": row["request_id"],
        "prompt_tokens": len(row["input_ids"]),
        "beams": beams,
    }


def _load_workload(
    path: Path, *, limit: int, context_len: int
) -> list[dict[str, Any]]:
    rows = read_jsonl(path, limit=limit)
    if len(rows) < limit:
        raise ValueError(f"{path} has {len(rows)} rows, expected {limit}")
    for line_number, row in enumerate(rows, start=1):
        input_ids = row.get("input_ids")
        if not isinstance(input_ids, list) or len(input_ids) != context_len:
            raise ValueError(
                f"{path}:{line_number} must contain {context_len} input_ids"
            )
    return rows


def _metric_summaries(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in (
        "wall_ms",
        "qps",
        "generated_tokens_per_s",
        "beam_candidates_per_s",
        "request_latency_ms_p50",
        "request_latency_ms_p95",
        "output_normalization_ms",
    ):
        samples = [run[field] for run in runs if run.get(field) is not None]
        result[f"{field}_samples"] = samples
        result[f"{field}_median"] = numeric_median(samples)
    return result


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, int(quantile * (len(sorted_values) - 1)))
    return sorted_values[index]


def _environment(torch: Any) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": sys.version,
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": cuda_available,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_device_capability": (
            torch.cuda.get_device_capability(0) if cuda_available else None
        ),
    }


def _git_commit(repo: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--vllm-repo", required=True)
    parser.add_argument("--workload-jsonl", required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=40000)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.context_len + args.decode_steps > args.max_model_len:
        raise ValueError("context length plus decode steps exceeds --max-model-len")
    if args.max_num_seqs < args.requests * args.beam_width:
        raise ValueError("--max-num-seqs cannot cover requests * beam width")
    required_tokens = args.requests * max(args.context_len, args.beam_width)
    if args.max_num_batched_tokens < required_tokens:
        raise ValueError(
            "--max-num-batched-tokens cannot cover the requested offline batch"
        )
    if args.warmup_runs < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    result = asyncio.run(run_benchmark(args))
    write_json(args.output_json, result)
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
