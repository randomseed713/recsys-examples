#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

three_way_wait_for_http_ready() {
  local url="$1"
  local timeout_s="${2:-300}"
  "${PYTHON_BIN:-python}" - "${url}" "${timeout_s}" <<'PY'
import sys
import time
import urllib.error
import urllib.request

url = sys.argv[1]
deadline = time.monotonic() + float(sys.argv[2])
last_error = None
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if 200 <= response.status < 300:
                raise SystemExit(0)
    except (OSError, urllib.error.URLError) as exc:
        last_error = exc
    time.sleep(1)
raise SystemExit(f"timed out waiting for {url}: {last_error!r}")
PY
}

three_way_stop_background_process() {
  local pid="${1:-}"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}
