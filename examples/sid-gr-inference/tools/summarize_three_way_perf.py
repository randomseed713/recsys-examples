# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize and summarize SID-GR, SGLang, and vLLM performance artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping

from tool_utils import optional_float, read_json, write_json

CASE_RE = re.compile(r"ctx(?P<context>\d+)_beam(?P<beam>\d+)_req(?P<requests>\d+)")


def build_offline_rows(root: Path) -> list[dict[str, Any]]:
    offline_dir = root / "offline"
    rows = []
    for gr_path in sorted(offline_dir.glob("gr_ctx*_beam*_req*.json")):
        match = CASE_RE.search(gr_path.stem)
        if match is None:
            continue
        suffix = match.group(0)
        paths = {
            "gr": gr_path,
            "sglang": offline_dir / f"sglang_{suffix}.json",
            "vllm": offline_dir / f"vllm_{suffix}.json",
        }
        if not all(path.exists() for path in paths.values()):
            continue
        artifacts = {name: read_json(path) for name, path in paths.items()}
        normalized_dir = offline_dir / "normalized"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        for framework, artifact in artifacts.items():
            write_json(
                normalized_dir / f"{framework}_{suffix}.json",
                _normalize_offline(
                    framework,
                    artifact,
                    context_len=int(match.group("context")),
                    beam_width=int(match.group("beam")),
                    requests=int(match.group("requests")),
                ),
            )
        walls = {
            framework: optional_float(artifact.get("wall_ms_median"))
            for framework, artifact in artifacts.items()
        }
        qps = {
            framework: _qps(framework, artifact, walls[framework])
            for framework, artifact in artifacts.items()
        }
        rows.append(
            {
                "mode": "offline",
                "context_len": int(match.group("context")),
                "beam_width": int(match.group("beam")),
                "batch_requests": int(match.group("requests")),
                **{f"{name}_wall_ms": walls[name] for name in walls},
                **{f"{name}_qps": qps[name] for name in qps},
                "sglang_over_gr": _ratio(walls["sglang"], walls["gr"]),
                "vllm_over_gr": _ratio(walls["vllm"], walls["gr"]),
                "vllm_over_sglang": _ratio(walls["vllm"], walls["sglang"]),
                "winner": min(
                    (name for name in walls if walls[name] is not None),
                    key=lambda name: walls[name],
                    default=None,
                ),
            }
        )
    return rows


def build_online_rows(root: Path) -> list[dict[str, Any]]:
    paths = {
        framework: root / "online" / f"{framework}.jsonl"
        for framework in ("gr", "sglang", "vllm")
    }
    if not all(path.exists() for path in paths.values()):
        return []
    artifacts = {name: _read_last_jsonl(path) for name, path in paths.items()}
    manifest = read_json(root / "manifest.json")
    walls = {
        framework: _duration_ms(artifact)
        for framework, artifact in artifacts.items()
    }
    qps = {
        framework: optional_float(artifact.get("request_throughput"))
        for framework, artifact in artifacts.items()
    }
    e2e_p50 = {
        framework: optional_float(artifact.get("median_e2e_latency_ms"))
        for framework, artifact in artifacts.items()
    }
    e2e_p99 = {
        framework: optional_float(artifact.get("p99_e2e_latency_ms"))
        for framework, artifact in artifacts.items()
    }
    sample = artifacts["gr"]
    return [
        {
            "mode": "online",
            "context_len": sample.get("random_input_len")
            or int(manifest["online_context_len"]),
            "beam_width": _extra_body_beam_width(sample)
            or int(manifest["online_beam_width"]),
            "batch_requests": sample.get("num_prompts")
            or sample.get("completed")
            or int(manifest["online_requests"]),
            **{f"{name}_wall_ms": walls[name] for name in walls},
            **{f"{name}_qps": qps[name] for name in qps},
            **{f"{name}_e2e_p50_ms": e2e_p50[name] for name in e2e_p50},
            **{f"{name}_e2e_p99_ms": e2e_p99[name] for name in e2e_p99},
            "sglang_over_gr": _ratio(walls["sglang"], walls["gr"]),
            "vllm_over_gr": _ratio(walls["vllm"], walls["gr"]),
            "vllm_over_sglang": _ratio(walls["vllm"], walls["sglang"]),
            "winner": max(
                (name for name in qps if qps[name] is not None),
                key=lambda name: qps[name],
                default=None,
            ),
        }
    ]


def _read_last_jsonl(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not isinstance(rows[-1], dict):
        raise ValueError(f"{path} does not contain a benchmark JSON object")
    return rows[-1]


def _duration_ms(artifact: Mapping[str, Any]) -> float | None:
    duration = optional_float(artifact.get("duration"))
    return duration * 1000.0 if duration is not None else None


def _extra_body_beam_width(artifact: Mapping[str, Any]) -> int | None:
    extra = artifact.get("extra_request_body")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            extra = None
    if not isinstance(extra, Mapping):
        return None
    sampling_params = extra.get("sampling_params")
    if isinstance(sampling_params, Mapping) and sampling_params.get("n") is not None:
        return int(sampling_params["n"])
    if extra.get("n") is not None:
        return int(extra["n"])
    return None


def _normalize_offline(
    framework: str,
    artifact: Mapping[str, Any],
    *,
    context_len: int,
    beam_width: int,
    requests: int,
) -> dict[str, Any]:
    if framework in {"sglang", "vllm"}:
        runs = artifact.get("runs") or []
    else:
        runs = [
            {
                "wall_ms": artifact.get("wall_ms_median"),
                "outputs": [
                    _normalize_gr_output(output)
                    for output in artifact.get("outputs") or []
                ],
            }
        ]
    return {
        "schema_version": "three_way_offline_v1",
        "framework": framework,
        "context_len": artifact.get("context_len") or context_len,
        "decode_steps": artifact.get("decode_steps")
        or _infer_decode_steps(framework, artifact),
        "beam_width": artifact.get("beam_width")
        or (artifact.get("engine_status") or {}).get("max_beam_width")
        or beam_width,
        "requests": artifact.get("requests")
        or artifact.get("responses")
        or requests,
        "wall_ms_median": artifact.get("wall_ms_median"),
        "qps_median": _qps(
            framework,
            artifact,
            optional_float(artifact.get("wall_ms_median")),
        ),
        "runs": runs,
    }


def _infer_decode_steps(framework: str, artifact: Mapping[str, Any]) -> int | None:
    if framework != "gr":
        return None
    outputs = artifact.get("outputs") or []
    if not outputs:
        return None
    candidates = outputs[0].get("beam_results") or []
    if not candidates:
        return None
    token_ids = candidates[0].get("output_ids")
    return len(token_ids) if isinstance(token_ids, list | tuple) else None


def _normalize_gr_output(output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "request_id": output.get("request_id"),
        "workload_id": output.get("workload_id"),
        "beams": [
            _normalize_candidate(candidate, rank)
            for rank, candidate in enumerate(output.get("beam_results") or [])
            if isinstance(candidate, Mapping)
        ],
    }


def _normalize_candidate(
    candidate: Mapping[str, Any], rank: int
) -> dict[str, Any]:
    token_ids = []
    for key in ("token_ids", "output_ids", "output_token_ids"):
        value = candidate.get(key)
        if isinstance(value, list | tuple):
            token_ids = [int(token_id) for token_id in value]
            break
    score = None
    for key in ("score", "sequence_score", "cumulative_score"):
        if candidate.get(key) is not None:
            score = float(candidate[key])
            break
    meta_info = candidate.get("meta_info")
    if score is None and isinstance(meta_info, Mapping):
        for key in ("sequence_score", "score", "cumulative_score"):
            if meta_info.get(key) is not None:
                score = float(meta_info[key])
                break
    return {"rank": rank, "token_ids": token_ids, "score": score}


def _qps(
    framework: str, artifact: Mapping[str, Any], wall_ms: float | None
) -> float | None:
    direct = optional_float(artifact.get("qps_median"))
    if direct is not None:
        return direct
    requests = artifact.get("requests") or artifact.get("responses")
    if framework == "gr" and wall_ms and requests:
        return float(requests) / (wall_ms / 1000.0)
    return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    return numerator / denominator if numerator and denominator else None


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "mode",
        "context_len",
        "beam_width",
        "batch_requests",
        "gr_wall_ms",
        "sglang_wall_ms",
        "vllm_wall_ms",
        "gr_qps",
        "sglang_qps",
        "vllm_qps",
        "gr_e2e_p50_ms",
        "sglang_e2e_p50_ms",
        "vllm_e2e_p50_ms",
        "gr_e2e_p99_ms",
        "sglang_e2e_p99_ms",
        "vllm_e2e_p99_ms",
        "sglang_over_gr",
        "vllm_over_gr",
        "vllm_over_sglang",
        "winner",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# SID-GR / SGLang / vLLM Performance",
        "",
        "| mode | ctx | beam | requests | GR ms | SGLang ms | vLLM ms | SGLang/GR | vLLM/GR | winner |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {context_len} | {beam_width} | {batch_requests} | "
            "{gr} | {sglang} | {vllm} | {sglang_ratio} | {vllm_ratio} | {winner} |".format(
                **row,
                gr=_fmt(row.get("gr_wall_ms")),
                sglang=_fmt(row.get("sglang_wall_ms")),
                vllm=_fmt(row.get("vllm_wall_ms")),
                sglang_ratio=_fmt(row.get("sglang_over_gr")),
                vllm_ratio=_fmt(row.get("vllm_over_gr")),
            )
        )
    online_rows = [row for row in rows if row["mode"] == "online"]
    if online_rows:
        row = online_rows[0]
        lines.extend(
            [
                "",
                "## Online request metrics",
                "",
                "| framework | request/s | median E2E ms | p99 E2E ms |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for framework in ("gr", "sglang", "vllm"):
            lines.append(
                "| {framework} | {qps} | {p50} | {p99} |".format(
                    framework=framework,
                    qps=_fmt(row.get(f"{framework}_qps")),
                    p50=_fmt(row.get(f"{framework}_e2e_p50_ms")),
                    p99=_fmt(row.get(f"{framework}_e2e_p99_ms")),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    number = optional_float(value)
    return "" if number is None else f"{number:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    rows = build_offline_rows(root) + build_online_rows(root)
    write_csv(rows, root / "summary.csv")
    write_markdown(rows, root / "summary.md")
    write_json(
        root / "summary.json",
        {
            "schema_version": "three_way_summary_v1",
            "manifest": "manifest.json",
            "rows": rows,
        },
    )
    print((root / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
