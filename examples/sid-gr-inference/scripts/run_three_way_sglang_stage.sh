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
MODEL_VARIANT="${MODEL_VARIANT:-Qwen3-1.7B}"
MODEL_DIR="${MODEL_DIR:-$(gr_default_model_dir "${MODEL_VARIANT}")}"
SGLANG_REPO="${SGLANG_REPO:-$(gr_default_sglang_repo)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT:-$(gr_default_decode_atten_root)}"
THREE_WAY_ROOT="${THREE_WAY_ROOT:-${SID_GR_ROOT}/benchmark_artifacts/three_way}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${THREE_WAY_ROOT}/${RUN_ID}"

CONTEXT_LENS="${CONTEXT_LENS:-1000 5000}"
BEAM_WIDTHS="${BEAM_WIDTHS:-256}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
GR_DECODE_STEPS="${GR_DECODE_STEPS:-2}"
BASELINE_DECODE_STEPS="${BASELINE_DECODE_STEPS:-3}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
REPEAT="${REPEAT:-3}"
RUN_ONLINE="${RUN_ONLINE:-1}"

ONLINE_CONTEXT_LEN="${ONLINE_CONTEXT_LEN:-5000}"
ONLINE_OUTPUT_LEN="${ONLINE_OUTPUT_LEN:-3}"
ONLINE_BEAM_WIDTH="${ONLINE_BEAM_WIDTH:-256}"
ONLINE_REQUESTS="${ONLINE_REQUESTS:-64}"
ONLINE_MAX_CONCURRENCY="${ONLINE_MAX_CONCURRENCY:-4}"
ONLINE_REQUEST_RATE="${ONLINE_REQUEST_RATE:-inf}"
ONLINE_WARMUP_REQUESTS="${ONLINE_WARMUP_REQUESTS:-0}"
ONLINE_RANDOM_SEED="${ONLINE_RANDOM_SEED:-1}"
GR_HOST="${GR_HOST:-127.0.0.1}"
GR_PORT="${GR_PORT:-8000}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-600}"

if [[ "${QUICK:-0}" == "1" ]]; then
  CONTEXT_LENS="${QUICK_CONTEXT_LENS:-1000}"
  BATCH_SIZES="${QUICK_BATCH_SIZES:-1}"
  WARMUP_RUNS="${QUICK_WARMUP_RUNS:-1}"
  REPEAT="${QUICK_REPEAT:-1}"
  ONLINE_REQUESTS="${QUICK_ONLINE_REQUESTS:-8}"
  ONLINE_MAX_CONCURRENCY="${QUICK_ONLINE_MAX_CONCURRENCY:-2}"
fi

[[ -x "${PYTHON_BIN}" ]] || { echo "missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -d "${MODEL_DIR}" ]] || { echo "missing model: ${MODEL_DIR}" >&2; exit 2; }
gr_require_sglang_repo "${SGLANG_REPO}" 1
if (( ONLINE_OUTPUT_LEN < 2 )); then
  echo "ONLINE_OUTPUT_LEN must be at least 2 for GR's decode_steps + 1 contract" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/offline" "${RUN_DIR}/online" "${RUN_DIR}/workloads" \
  "${RUN_DIR}/logs" "${RUN_DIR}/environment"
printf '%s\n' "${RUN_ID}" >"${THREE_WAY_ROOT}/LATEST"
gr_setup_local_cache_env

for context_len in ${CONTEXT_LENS}; do
  for requests in ${BATCH_SIZES}; do
    workload="${RUN_DIR}/workloads/qwen3_ctx${context_len}_req${requests}.jsonl"
    PYTHONPATH="${SID_GR_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" \
      "${SID_GR_ROOT}/tools/make_qwen3_beam_workload.py" \
      --model-dir "${MODEL_DIR}" \
      --context-len "${context_len}" \
      --requests "${requests}" \
      --no-tokenizer \
      --output-jsonl "${workload}"
  done
done
(
  cd "${RUN_DIR}"
  find workloads -type f -name '*.jsonl' -print0 | sort -z | xargs -0 sha256sum \
    >workloads.sha256
)

SGLANG_COMMIT="$(git -C "${SGLANG_REPO}" rev-parse HEAD)"
RUN_DIR="${RUN_DIR}" MODEL_DIR="${MODEL_DIR}" SGLANG_REPO="${SGLANG_REPO}" \
SGLANG_COMMIT="${SGLANG_COMMIT}" CONTEXT_LENS="${CONTEXT_LENS}" \
BEAM_WIDTHS="${BEAM_WIDTHS}" BATCH_SIZES="${BATCH_SIZES}" \
GR_DECODE_STEPS="${GR_DECODE_STEPS}" BASELINE_DECODE_STEPS="${BASELINE_DECODE_STEPS}" \
WARMUP_RUNS="${WARMUP_RUNS}" REPEAT="${REPEAT}" \
RUN_ONLINE="${RUN_ONLINE}" \
ONLINE_CONTEXT_LEN="${ONLINE_CONTEXT_LEN}" ONLINE_OUTPUT_LEN="${ONLINE_OUTPUT_LEN}" \
ONLINE_BEAM_WIDTH="${ONLINE_BEAM_WIDTH}" ONLINE_REQUESTS="${ONLINE_REQUESTS}" \
ONLINE_MAX_CONCURRENCY="${ONLINE_MAX_CONCURRENCY}" \
ONLINE_REQUEST_RATE="${ONLINE_REQUEST_RATE}" ONLINE_WARMUP_REQUESTS="${ONLINE_WARMUP_REQUESTS}" \
ONLINE_RANDOM_SEED="${ONLINE_RANDOM_SEED}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

keys = (
    "MODEL_DIR", "SGLANG_REPO", "SGLANG_COMMIT", "CONTEXT_LENS",
    "BEAM_WIDTHS", "BATCH_SIZES", "GR_DECODE_STEPS",
    "BASELINE_DECODE_STEPS", "WARMUP_RUNS", "REPEAT",
    "RUN_ONLINE",
    "ONLINE_CONTEXT_LEN", "ONLINE_OUTPUT_LEN", "ONLINE_BEAM_WIDTH",
    "ONLINE_REQUESTS", "ONLINE_MAX_CONCURRENCY", "ONLINE_REQUEST_RATE",
    "ONLINE_WARMUP_REQUESTS", "ONLINE_RANDOM_SEED",
)
manifest = {"schema_version": "three_way_run_v1"}
manifest.update({key.lower(): os.environ[key] for key in keys})
manifest["vllm_project"] = "Theta/vllm"
manifest["vllm_branch"] = "rec_vllm_0190_1"
manifest["vllm_commit"] = "2c206276111fc3217ee76f9bcc16f46422d4298b"
path = Path(os.environ["RUN_DIR"]) / "manifest.json"
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

"${PYTHON_BIN}" -m pip freeze >"${RUN_DIR}/environment/sglang-pip-freeze.txt"
git -C "${SID_GR_ROOT}" rev-parse HEAD >"${RUN_DIR}/environment/sid-gr-commit.txt"
printf '%s\n' "${SGLANG_COMMIT}" >"${RUN_DIR}/environment/sglang-commit.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q >"${RUN_DIR}/environment/sglang-nvidia-smi.txt"
fi

for context_len in ${CONTEXT_LENS}; do
  for beam_width in ${BEAM_WIDTHS}; do
    for requests in ${BATCH_SIZES}; do
      suffix="ctx${context_len}_beam${beam_width}_req${requests}"
      workload="${RUN_DIR}/workloads/qwen3_ctx${context_len}_req${requests}.jsonl"
      echo "== offline ${suffix} =="
      PYTHONPATH="${SGLANG_REPO}/python:${SID_GR_ROOT}/src:${PYTHONPATH:-}" \
      GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT}" "${PYTHON_BIN}" \
        "${SID_GR_ROOT}/tools/run_qwen3_real_weight_serving.py" \
        --model-dir "${MODEL_DIR}" \
        --workload-jsonl "${workload}" \
        --context-len "${context_len}" \
        --decode-steps "${GR_DECODE_STEPS}" \
        --beam-width "${beam_width}" \
        --requests "${requests}" \
        --max-batch-size "${requests}" \
        --batched-decode --continuous --decode-backend real --device cuda \
        --beam-kv-pool-capacity "${requests}" \
        --context-kv-pool-capacity "${requests}" \
        --record-outputs \
        --warmup-runs "${WARMUP_RUNS}" --repeat "${REPEAT}" \
        --output-json "${RUN_DIR}/offline/gr_${suffix}.json" \
        >"${RUN_DIR}/logs/gr_${suffix}.log" 2>&1

      PYTHONPATH="${SGLANG_REPO}/python:${SID_GR_ROOT}/src:${PYTHONPATH:-}" \
      "${PYTHON_BIN}" "${SID_GR_ROOT}/tools/run_sglang_beam_benchmark.py" \
        --model-dir "${MODEL_DIR}" \
        --sglang-repo "${SGLANG_REPO}" \
        --workload-jsonl "${workload}" \
        --context-len "${context_len}" \
        --decode-steps "${BASELINE_DECODE_STEPS}" \
        --beam-width "${beam_width}" \
        --requests "${requests}" \
        --arrival-mode batch \
        --disable-radix-cache \
        --warmup-runs "${WARMUP_RUNS}" --repeat "${REPEAT}" \
        --use-input-ids --no-tokenizer \
        --output-json "${RUN_DIR}/offline/sglang_${suffix}.json" \
        >"${RUN_DIR}/logs/sglang_${suffix}.log" 2>&1
    done
  done
done

if [[ "${RUN_ONLINE}" == "1" ]]; then
  EXTRA_REQUEST_BODY="$(gr_beam_sampling_extra_request_body "${ONLINE_OUTPUT_LEN}" "${ONLINE_BEAM_WIDTH}")"
  server_pid=""
  trap 'three_way_stop_background_process "${server_pid}"' EXIT

  GR_HTTP_HOST=0.0.0.0 GR_HTTP_PORT="${GR_PORT}" \
  GR_CONTEXT_LEN="${ONLINE_CONTEXT_LEN}" \
  GR_DECODE_STEPS="$((ONLINE_OUTPUT_LEN - 1))" \
  GR_BEAM_WIDTH="${ONLINE_BEAM_WIDTH}" \
  GR_MAX_BATCH_SIZE="${ONLINE_MAX_CONCURRENCY}" \
  GR_BEAM_KV_POOL_CAPACITY="${ONLINE_MAX_CONCURRENCY}" \
  GR_CONTEXT_KV_POOL_CAPACITY="${ONLINE_MAX_CONCURRENCY}" \
  MODEL_DIR="${MODEL_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    "${SID_GR_ROOT}/scripts/serve_qwen3_gr_http.sh" \
    >"${RUN_DIR}/logs/gr-online-server.log" 2>&1 &
  server_pid=$!
  three_way_wait_for_http_ready "http://${GR_HOST}:${GR_PORT}/health" "${SERVER_READY_TIMEOUT_S}"
  PYTHONPATH="${SGLANG_REPO}/python:${SID_GR_ROOT}/src:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -m sglang.bench_serving \
    --backend sglang --host "${GR_HOST}" --port "${GR_PORT}" \
    --model "${MODEL_DIR}" --tokenizer "${MODEL_DIR}" \
    --dataset-name random --random-input-len "${ONLINE_CONTEXT_LEN}" \
    --random-output-len "${ONLINE_OUTPUT_LEN}" --random-range-ratio 1 \
    --num-prompts "${ONLINE_REQUESTS}" --request-rate "${ONLINE_REQUEST_RATE}" \
    --max-concurrency "${ONLINE_MAX_CONCURRENCY}" \
    --warmup-requests "${ONLINE_WARMUP_REQUESTS}" --seed "${ONLINE_RANDOM_SEED}" \
    --disable-stream --disable-ignore-eos --tokenize-prompt \
    --extra-request-body "${EXTRA_REQUEST_BODY}" --output-details \
    --output-file "${RUN_DIR}/online/gr.jsonl"
  three_way_stop_background_process "${server_pid}"
  server_pid=""

  PYTHONPATH="${SGLANG_REPO}/python:${SID_GR_ROOT}/src:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -m sglang.launch_server \
    --model-path "${MODEL_DIR}" --host 0.0.0.0 --port "${SGLANG_PORT}" \
    --enable-beam-search --disable-radix-cache \
    >"${RUN_DIR}/logs/sglang-online-server.log" 2>&1 &
  server_pid=$!
  three_way_wait_for_http_ready "http://${SGLANG_HOST}:${SGLANG_PORT}/health" "${SERVER_READY_TIMEOUT_S}"
  PYTHONPATH="${SGLANG_REPO}/python:${SID_GR_ROOT}/src:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -m sglang.bench_serving \
    --backend sglang --host "${SGLANG_HOST}" --port "${SGLANG_PORT}" \
    --model "${MODEL_DIR}" --tokenizer "${MODEL_DIR}" \
    --dataset-name random --random-input-len "${ONLINE_CONTEXT_LEN}" \
    --random-output-len "${ONLINE_OUTPUT_LEN}" --random-range-ratio 1 \
    --num-prompts "${ONLINE_REQUESTS}" --request-rate "${ONLINE_REQUEST_RATE}" \
    --max-concurrency "${ONLINE_MAX_CONCURRENCY}" \
    --warmup-requests "${ONLINE_WARMUP_REQUESTS}" --seed "${ONLINE_RANDOM_SEED}" \
    --disable-stream --disable-ignore-eos --tokenize-prompt \
    --extra-request-body "${EXTRA_REQUEST_BODY}" --output-details \
    --output-file "${RUN_DIR}/online/sglang.jsonl"
  three_way_stop_background_process "${server_pid}"
  server_pid=""
  trap - EXIT
fi

echo "SGLang stage complete"
echo "RUN_ID=${RUN_ID}"
echo "RUN_DIR=${RUN_DIR}"
