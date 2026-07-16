#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common_paths.sh
source "${SCRIPT_DIR}/common_paths.sh"

PERSIST_ROOT="${PERSIST_ROOT:-${GR_INFERENCE_WORKSPACE_ROOT}}"
VLLM_REPO="${VLLM_REPO:-${PERSIST_ROOT}/vllm_rec_beam}"
VLLM_PROJECT="${VLLM_PROJECT:-Theta/vllm}"
VLLM_BRANCH="${VLLM_BRANCH:-rec_vllm_0190_1}"
VLLM_COMMIT="${VLLM_COMMIT:-2c206276111fc3217ee76f9bcc16f46422d4298b}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://artifacts.antgroup-inc.cn/simple/}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
BUILD_ID="${BUILD_ID:-$(date +%Y%m%d_%H%M%S)}"
BUILD_DIR="${VLLM_BUILD_LOG_ROOT:-${PERSIST_ROOT}/benchmark_artifacts/three_way/build}/${BUILD_ID}"
FETCHCONTENT_BASE_DIR="${FETCHCONTENT_BASE_DIR:-${PERSIST_ROOT}/.cache/vllm-fetchcontent}"
MAX_JOBS="${MAX_JOBS:-8}"
NVCC_THREADS="${NVCC_THREADS:-2}"

mkdir -p "${BUILD_DIR}" "${FETCHCONTENT_BASE_DIR}" "$(dirname "${VLLM_REPO}")"

for command in antcode git gcc g++; do
  command -v "${command}" >/dev/null 2>&1 || {
    echo "required command is missing: ${command}" >&2
    exit 2
  }
done
antcode --version | tee "${BUILD_DIR}/antcode-version.txt"
[[ -x "${PYTHON_BIN}" ]] || {
  echo "Python does not exist or is not executable: ${PYTHON_BIN}" >&2
  exit 2
}
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || {
  echo "CUDA Toolkit nvcc is missing: ${CUDA_HOME}/bin/nvcc" >&2
  exit 2
}
"${CUDA_HOME}/bin/nvcc" --version | tee "${BUILD_DIR}/nvcc-version.txt"
grep -q "release 12\.8" "${BUILD_DIR}/nvcc-version.txt" || {
  echo "native vLLM build requires nvcc 12.8" >&2
  exit 2
}

"${PYTHON_BIN}" - <<'PY'
import sys

import torch

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected Python 3.10, got {sys.version}")
if torch.__version__.split("+", 1)[0] != "2.10.0":
    raise SystemExit(f"expected torch 2.10.0, got {torch.__version__}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected torch CUDA 12.8, got {torch.version.cuda}")
print("python", sys.version)
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
PY

"${PYTHON_BIN}" -m pip freeze >"${BUILD_DIR}/pip-freeze-before.txt"

if [[ -e "${VLLM_REPO}" && ! -d "${VLLM_REPO}/.git" ]]; then
  echo "VLLM_REPO exists but is not a git checkout: ${VLLM_REPO}" >&2
  exit 2
fi
if [[ ! -d "${VLLM_REPO}/.git" ]]; then
  antcode clone "${VLLM_PROJECT}" "${VLLM_REPO}" \
    --branch "${VLLM_BRANCH}" \
    --commit "${VLLM_COMMIT}" \
    --depth 1
fi
if [[ -n "$(git -C "${VLLM_REPO}" status --porcelain)" ]]; then
  echo "VLLM_REPO must be clean: ${VLLM_REPO}" >&2
  exit 2
fi
actual_commit="$(git -C "${VLLM_REPO}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${VLLM_COMMIT}" ]]; then
  echo "VLLM_REPO commit mismatch: expected ${VLLM_COMMIT}, got ${actual_commit}" >&2
  exit 2
fi
origin_url="$(git -C "${VLLM_REPO}" remote get-url origin)"
if [[ ! "${origin_url}" =~ [:/]Theta/vllm(\.git)?$ ]]; then
  echo "VLLM_REPO origin is not Theta/vllm: ${origin_url}" >&2
  exit 2
fi

"${PYTHON_BIN}" -m pip install \
  --index-url "${PIP_INDEX_URL}" \
  --no-deps \
  "cmake>=3.26.1" \
  ninja \
  "packaging>=24.2" \
  "setuptools>=77.0.3,<81.0.0" \
  "setuptools-scm>=8.0" \
  wheel \
  "jinja2>=3.1.6" \
  2>&1 | tee "${BUILD_DIR}/build-dependencies.log"

command -v cmake >/dev/null 2>&1 || {
  echo "cmake is unavailable after build dependency setup" >&2
  exit 2
}
command -v ninja >/dev/null 2>&1 || {
  echo "ninja is unavailable after build dependency setup" >&2
  exit 2
}
cmake --version | tee "${BUILD_DIR}/cmake-version.txt"
ninja --version | tee "${BUILD_DIR}/ninja-version.txt"

env \
  -u VLLM_USE_PRECOMPILED \
  -u VLLM_PRECOMPILED_WHEEL_LOCATION \
  -u VLLM_PRECOMPILED_WHEEL_COMMIT \
  -u VLLM_PRECOMPILED_WHEEL_VARIANT \
  CUDA_HOME="${CUDA_HOME}" \
  TORCH_CUDA_ARCH_LIST=8.9 \
  CMAKE_BUILD_TYPE=Release \
  MAX_JOBS="${MAX_JOBS}" \
  NVCC_THREADS="${NVCC_THREADS}" \
  FETCHCONTENT_BASE_DIR="${FETCHCONTENT_BASE_DIR}" \
  "${PYTHON_BIN}" -m pip install \
    --no-deps \
    --no-build-isolation \
    -e "${VLLM_REPO}" \
    2>&1 | tee "${BUILD_DIR}/vllm-native-build.log"

VLLM_REPO="${VLLM_REPO}" VLLM_COMMIT="${VLLM_COMMIT}" "${PYTHON_BIN}" - <<'PY' \
  | tee "${BUILD_DIR}/vllm-import-check.txt"
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import vllm
import vllm._C
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.rec_beam.rec_beam_worker import RecBeamWorker

repo = Path(os.environ["VLLM_REPO"]).resolve()
package = Path(vllm.__file__).resolve()
if not package.is_relative_to(repo):
    raise SystemExit(f"vLLM is not loaded from editable source: {package}")
if not hasattr(AsyncLLM, "rec_beam_search"):
    raise SystemExit("AsyncLLM.rec_beam_search is missing")
print("vllm version", importlib.metadata.version("vllm"))
print("vllm package", package)
print("vllm native", vllm._C.__file__)
print("rec beam worker", RecBeamWorker.__module__)
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
direct_url = importlib.metadata.distribution("vllm").read_text("direct_url.json")
print("direct_url", json.loads(direct_url) if direct_url else None)
PY

"${PYTHON_BIN}" -m pip check | tee "${BUILD_DIR}/pip-check.txt"
"${PYTHON_BIN}" -m pip freeze >"${BUILD_DIR}/pip-freeze-after.txt"

echo "vLLM native editable build complete"
echo "repo: ${VLLM_REPO}"
echo "commit: ${VLLM_COMMIT}"
echo "logs: ${BUILD_DIR}"
