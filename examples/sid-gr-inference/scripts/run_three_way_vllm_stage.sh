#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common_paths.sh
source "${SCRIPT_DIR}/common_paths.sh"
# shellcheck source=scripts/three_way_common.sh
source "${SCRIPT_DIR}/three_way_common.sh"

SID_GR_ROOT="${SID_GR_ROOT:-${GR_INFERENCE_REPO_ROOT}}"
PERSIST_ROOT="${PERSIST_ROOT:-${GR_INFERENCE_WORKSPACE_ROOT}}"
MODEL_VARIANT="${MODEL_VARIANT:-Qwen3-1.7B}"
MODEL_DIR="${MODEL_DIR:-$(gr_default_model_dir "${MODEL_VARIANT}")}"
SGLANG_REPO="${SGLANG_REPO:-$(gr_default_sglang_repo)}"
VLLM_REPO="${VLLM_REPO:-${PERSIST_ROOT}/vllm_rec_beam}"
VLLM_COMMIT="${VLLM_COMMIT:-2c206276111fc3217ee76f9bcc16f46422d4298b}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
VLLM_BIN="${VLLM_BIN:-/opt/conda/bin/vllm}"
THREE_WAY_ROOT="${THREE_WAY_ROOT:-${SID_GR_ROOT}/benchmark_artifacts/three_way}"
if [[ -z "${RUN_ID:-}" && -f "${THREE_WAY_ROOT}/LATEST" ]]; then
  RUN_ID="$(<"${THREE_WAY_ROOT}/LATEST")"
fi
RUN_ID="${RUN_ID:?set RUN_ID or run the SGLang stage first}"
RUN_DIR="${THREE_WAY_ROOT}/${RUN_ID}"

manifest_value() {
  "${PYTHON_BIN}" - "${RUN_DIR}/manifest.json" "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
value = manifest.get(sys.argv[2])
if value is None:
    raise SystemExit(f"manifest key is missing: {sys.argv[2]}")
print(value)
PY
}

CONTEXT_LENS="$(manifest_value context_lens)"
BEAM_WIDTHS="$(manifest_value beam_widths)"
BATCH_SIZES="$(manifest_value batch_sizes)"
BASELINE_DECODE_STEPS="$(manifest_value baseline_decode_steps)"
WARMUP_RUNS="$(manifest_value warmup_runs)"
REPEAT="$(manifest_value repeat)"
VLLM_OFFLINE_MAX_NUM_SEQS="${VLLM_OFFLINE_MAX_NUM_SEQS:-2048}"
VLLM_OFFLINE_MAX_NUM_BATCHED_TOKENS="${VLLM_OFFLINE_MAX_NUM_BATCHED_TOKENS:-40000}"
RUN_ONLINE="$(manifest_value run_online)"

ONLINE_CONTEXT_LEN="$(manifest_value online_context_len)"
ONLINE_OUTPUT_LEN="$(manifest_value online_output_len)"
ONLINE_BEAM_WIDTH="$(manifest_value online_beam_width)"
ONLINE_REQUESTS="$(manifest_value online_requests)"
ONLINE_MAX_CONCURRENCY="$(manifest_value online_max_concurrency)"
ONLINE_REQUEST_RATE="$(manifest_value online_request_rate)"
ONLINE_WARMUP_REQUESTS="$(manifest_value online_warmup_requests)"
ONLINE_RANDOM_SEED="$(manifest_value online_random_seed)"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_ONLINE_MAX_NUM_SEQS="${VLLM_ONLINE_MAX_NUM_SEQS:-$((ONLINE_BEAM_WIDTH * ONLINE_MAX_CONCURRENCY))}"
VLLM_ONLINE_MAX_NUM_BATCHED_TOKENS="${VLLM_ONLINE_MAX_NUM_BATCHED_TOKENS:-$((ONLINE_CONTEXT_LEN * ONLINE_MAX_CONCURRENCY))}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-600}"

[[ -d "${RUN_DIR}" ]] || { echo "missing run directory: ${RUN_DIR}" >&2; exit 2; }
[[ -d "${MODEL_DIR}" ]] || { echo "missing model: ${MODEL_DIR}" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -x "${VLLM_BIN}" ]] || { echo "missing vLLM CLI: ${VLLM_BIN}" >&2; exit 2; }
gr_require_sglang_repo "${SGLANG_REPO}" 1
manifest_model_dir="$(manifest_value model_dir)"
if [[ "${MODEL_DIR}" != "${manifest_model_dir}" ]]; then
  echo "MODEL_DIR differs from the SGLang-stage manifest" >&2
  echo "manifest: ${manifest_model_dir}" >&2
  echo "current:  ${MODEL_DIR}" >&2
  exit 2
fi
expected_sglang_commit="$(manifest_value sglang_commit)"
actual_sglang_commit="$(git -C "${SGLANG_REPO}" rev-parse HEAD)"
if [[ "${actual_sglang_commit}" != "${expected_sglang_commit}" ]]; then
  echo "SGLang commit differs from the run manifest" >&2
  echo "expected: ${expected_sglang_commit}" >&2
  echo "actual:   ${actual_sglang_commit}" >&2
  exit 2
fi
actual_commit="$(git -C "${VLLM_REPO}" rev-parse HEAD)"
[[ "${actual_commit}" == "${VLLM_COMMIT}" ]] || {
  echo "vLLM commit mismatch: expected ${VLLM_COMMIT}, got ${actual_commit}" >&2
  exit 2
}
(
  cd "${RUN_DIR}"
  sha256sum -c workloads.sha256
)

VLLM_REPO="${VLLM_REPO}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import vllm
import vllm._C
from vllm.v1.engine.async_llm import AsyncLLM

repo = Path(os.environ["VLLM_REPO"]).resolve()
if not Path(vllm.__file__).resolve().is_relative_to(repo):
    raise SystemExit(f"vLLM is not loaded from {repo}: {vllm.__file__}")
if not hasattr(AsyncLLM, "rec_beam_search"):
    raise SystemExit("AsyncLLM.rec_beam_search is missing")
print("vllm", vllm.__file__)
print("vllm._C", vllm._C.__file__)
PY

mkdir -p "${RUN_DIR}/offline" "${RUN_DIR}/online" "${RUN_DIR}/logs" \
  "${RUN_DIR}/environment"
"${PYTHON_BIN}" -m pip freeze >"${RUN_DIR}/environment/vllm-pip-freeze.txt"
printf '%s\n' "${actual_commit}" >"${RUN_DIR}/environment/vllm-commit.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q >"${RUN_DIR}/environment/vllm-nvidia-smi.txt"
fi

for context_len in ${CONTEXT_LENS}; do
  for beam_width in ${BEAM_WIDTHS}; do
    for requests in ${BATCH_SIZES}; do
      suffix="ctx${context_len}_beam${beam_width}_req${requests}"
      workload="${RUN_DIR}/workloads/qwen3_ctx${context_len}_req${requests}.jsonl"
      echo "== vLLM offline ${suffix} =="
      "${PYTHON_BIN}" "${SID_GR_ROOT}/tools/run_vllm_rec_beam_benchmark.py" \
        --model-dir "${MODEL_DIR}" --vllm-repo "${VLLM_REPO}" \
        --workload-jsonl "${workload}" --context-len "${context_len}" \
        --decode-steps "${BASELINE_DECODE_STEPS}" --beam-width "${beam_width}" \
        --requests "${requests}" --warmup-runs "${WARMUP_RUNS}" --repeat "${REPEAT}" \
        --max-model-len 8192 \
        --max-num-seqs "${VLLM_OFFLINE_MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${VLLM_OFFLINE_MAX_NUM_BATCHED_TOKENS}" \
        --output-json "${RUN_DIR}/offline/vllm_${suffix}.json" \
        >"${RUN_DIR}/logs/vllm_${suffix}.log" 2>&1
    done
  done
done

if [[ "${RUN_ONLINE}" == "1" ]]; then
  VLLM_EXTRA_REQUEST_BODY="$("${PYTHON_BIN}" - "${ONLINE_OUTPUT_LEN}" "${ONLINE_BEAM_WIDTH}" <<'PY'
import json
import sys

print(json.dumps({
    "use_beam_search": True,
    "n": int(sys.argv[2]),
    "max_tokens": int(sys.argv[1]),
    "temperature": 0.0,
    "ignore_eos": True,
    "length_penalty": 1.0,
    "stream": False,
    "echo": False,
    "rec_only_return_generate_token_ids": True,
}))
PY
)"
  server_pid=""
  trap 'three_way_stop_background_process "${server_pid}"' EXIT
  "${VLLM_BIN}" serve "${MODEL_DIR}" \
    --host 0.0.0.0 --port "${VLLM_PORT}" \
    --dtype bfloat16 --tensor-parallel-size 1 --pipeline-parallel-size 1 \
    --max-model-len 8192 \
    --max-num-seqs "${VLLM_ONLINE_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${VLLM_ONLINE_MAX_NUM_BATCHED_TOKENS}" \
    --no-enable-prefix-caching --rec-use-fast-beam-search \
    >"${RUN_DIR}/logs/vllm-online-server.log" 2>&1 &
  server_pid=$!
  three_way_wait_for_http_ready "http://${VLLM_HOST}:${VLLM_PORT}/health" "${SERVER_READY_TIMEOUT_S}"
  PYTHONPATH="${SGLANG_REPO}/python:${SID_GR_ROOT}/src:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -m sglang.bench_serving \
    --backend vllm --host "${VLLM_HOST}" --port "${VLLM_PORT}" \
    --model "${MODEL_DIR}" --tokenizer "${MODEL_DIR}" \
    --dataset-name random --random-input-len "${ONLINE_CONTEXT_LEN}" \
    --random-output-len "${ONLINE_OUTPUT_LEN}" --random-range-ratio 1 \
    --num-prompts "${ONLINE_REQUESTS}" --request-rate "${ONLINE_REQUEST_RATE}" \
    --max-concurrency "${ONLINE_MAX_CONCURRENCY}" \
    --warmup-requests "${ONLINE_WARMUP_REQUESTS}" --seed "${ONLINE_RANDOM_SEED}" \
    --disable-stream --disable-ignore-eos \
    --extra-request-body "${VLLM_EXTRA_REQUEST_BODY}" --output-details \
    --output-file "${RUN_DIR}/online/vllm.jsonl"
  three_way_stop_background_process "${server_pid}"
  server_pid=""
  trap - EXIT
fi

"${PYTHON_BIN}" "${SID_GR_ROOT}/tools/summarize_three_way_perf.py" "${RUN_DIR}"
echo "vLLM stage complete: ${RUN_DIR}"
