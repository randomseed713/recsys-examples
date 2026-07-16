---
name: sid-gr-three-way-benchmark
description: 帮助用户在 SGLang 和 vLLM 两个现有环境中快速运行 examples/sid-gr-inference 的 SID-GR、SGLang 与 vLLM REC_BEAM 三方 benchmark。先根据当前环境已安装的 SGLang 或 vLLM 自动选择运行阶段，再衔接 RUN_ID、共享目录和结果汇总。
---

# SID-GR / SGLang / vLLM 三方性能对比

本 Skill 的首要目标是让用户快速完成一次三方 benchmark：先在 SGLang
环境运行 SID-GR 和 SGLang，再切到 vLLM 环境完成原生编译、vLLM 测量和
三方汇总。优先给出可直接复制的命令，只有命令失败时才展开编译细节和
排障说明。

## 面向用户的操作方式

### 先识别当前环境

收到“运行 benchmark”之类的请求后，先用当前环境的
`/opt/conda/bin/python` 检查包：

```bash
/opt/conda/bin/python - <<'PY'
import importlib.util
import json

print(json.dumps({
    "sglang": importlib.util.find_spec("sglang") is not None,
    "vllm": importlib.util.find_spec("vllm") is not None,
}, sort_keys=True))
PY
```

按下面的规则直接执行，不要先让用户选择环境：

1. 检测到 SGLang：当前是 SGLang 环境，运行 SID-GR + SGLang 第一阶段。
2. 否则检测到 vLLM：当前是 vLLM 环境，运行 vLLM bootstrap 检查和
   第二阶段。

不要为了检测环境临时设置 SGLang `PYTHONPATH`，否则会把 vLLM 环境误判
为 SGLang 环境。环境判断完成后，运行 SGLang benchmark 客户端时才设置
目标源码的 `PYTHONPATH`。

### 检测到 SGLang 环境

直接运行第一阶段。用户没有指定 `RUN_ID` 时让脚本生成，并把脚本输出的
`RUN_ID`、`RUN_DIR` 告诉用户：

```bash
cd examples/sid-gr-inference

MODEL_DIR=/path/to/Qwen3-1.7B \
SGLANG_REPO=/path/on/persistent-disk/sglang_beam_search \
QUICK=1 \
scripts/run_three_way_sglang_stage.sh
```

这个环境只运行 SID-GR 和 SGLang。完成后让用户切换到 vLLM 环境，不在
当前环境安装或运行 vLLM。

### 检测到 vLLM 环境

先检查共享目录中是否有第一阶段写入的
`benchmark_artifacts/three_way/LATEST` 和对应 `manifest.json`。没有时
停止，告诉用户需要先在 SGLang 环境运行第一阶段。

随后检查当前 vLLM 是否已经从目标 REC_BEAM checkout 以 editable 方式
加载：

```bash
/opt/conda/bin/python - <<'PY'
import vllm
import vllm._C
from vllm.v1.engine.async_llm import AsyncLLM

print(vllm.__file__)
print(vllm._C.__file__)
print(hasattr(AsyncLLM, "rec_beam_search"))
PY
```

如果当前还是镜像自带 wheel，或缺少 `rec_beam_search`，执行一次：

```bash
PERSIST_ROOT=/path/on/persistent-disk \
scripts/bootstrap_three_way_vllm.sh
```

确认原生 editable 安装后，直接读取 `LATEST` 和 manifest 运行第二阶段：

```bash
PERSIST_ROOT=/path/on/persistent-disk \
MODEL_DIR=/path/to/Qwen3-1.7B \
SGLANG_REPO=/path/on/persistent-disk/sglang_beam_search \
scripts/run_three_way_vllm_stage.sh
```

第二阶段结束后返回 `summary.md`、`summary.csv`、`summary.json` 的实际
路径。不要让用户再手工输入第一阶段已经写入 manifest 的 workload 参数。

开始前先确认或从当前上下文取得下面四个值：

| 变量 | 含义 |
| --- | --- |
| `RUN_ID` | 两个环境共用的运行编号 |
| `MODEL_DIR` | 共享持久盘上的模型 snapshot |
| `SGLANG_REPO` | 共享持久盘上的 SGLang 目标分支源码 |
| `PERSIST_ROOT` | 两个环境都能访问的持久盘根目录 |

如果用户没有指定 `RUN_ID`，生成一个带时间戳的值，并要求两个环境使用
同一个值。模型和源码路径能从当前环境确认时直接使用，不重复询问。

默认先提供 quick 流程。用户明确要求完整数据时，再去掉 `QUICK=1`。

### 第一步：SGLang 环境

让用户进入 SGLang 环境，在 `examples/sid-gr-inference` 下执行：

```bash
RUN_ID=three_way_quick \
MODEL_DIR=/path/to/Qwen3-1.7B \
SGLANG_REPO=/path/on/persistent-disk/sglang_beam_search \
QUICK=1 \
scripts/run_three_way_sglang_stage.sh
```

第一阶段成功后，提醒用户保留输出中的 `RUN_ID` 和 `RUN_DIR`，然后切换到
vLLM 环境。不要让用户在 SGLang 环境安装 vLLM。

### 第二步：vLLM 环境

如果这个 vLLM checkout 尚未完成原生 editable 编译，先执行：

```bash
PERSIST_ROOT=/path/on/persistent-disk \
scripts/bootstrap_three_way_vllm.sh
```

编译已完成且 `vllm.__file__` 指向目标源码时，不重复编译。随后执行：

```bash
RUN_ID=three_way_quick \
PERSIST_ROOT=/path/on/persistent-disk \
MODEL_DIR=/path/to/Qwen3-1.7B \
SGLANG_REPO=/path/on/persistent-disk/sglang_beam_search \
scripts/run_three_way_vllm_stage.sh
```

第二阶段会从第一阶段的 manifest 读取 quick/full 配置，不需要再次传
`QUICK=1`。成功后直接告诉用户最终报告路径：

```text
examples/sid-gr-inference/benchmark_artifacts/three_way/${RUN_ID}/summary.md
```

如果命令失败，先返回对应日志路径和最后一段错误，再参考后面的常见问题
排查。不要一开始就向用户展开所有构建参数。

## 必须遵守的约束

- SID-GR/SGLang 与 vLLM 使用两个现有环境，通过同一块持久盘交换代码、
  模型和结果。
- 不创建新镜像。
- 不为这个 benchmark 修改
  `examples/sid-gr-inference/README.md` 或
  `examples/sid-gr-inference/scripts/common_paths.sh`。
- 不重新安装 SGLang。运行时通过
  `PYTHONPATH="${SGLANG_REPO}/python:${SID_GR_ROOT}/src"`
  使用目标分支源码。
- 不用 `PYTHONPATH` 覆盖 vLLM。vLLM 必须以原生 editable 方式安装。
- 不使用 `VLLM_USE_PRECOMPILED`、
  `VLLM_PRECOMPILED_WHEEL_LOCATION` 或任何预编译 wheel 回退。
- vLLM 固定为 AntCode 项目 `Theta/vllm`、分支
  `rec_vllm_0190_1`、提交
  `2c206276111fc3217ee76f9bcc16f46422d4298b`。
- 只使用一张 L20，TP、PP、DP 都设为 1。
- 三套在线压测统一使用 `sglang.bench_serving`。GR 和 SGLang 使用
  `sglang` backend，vLLM 使用 `vllm` backend。
- SGLang 阶段生成的 `manifest.json` 是本次运行的唯一配置来源。
  vLLM 阶段必须读取同一份离线矩阵和在线参数。

## 代码入口

| 用途 | 路径 |
| --- | --- |
| vLLM 原生编译 | `examples/sid-gr-inference/scripts/bootstrap_three_way_vllm.sh` |
| GR 和 SGLang 阶段 | `examples/sid-gr-inference/scripts/run_three_way_sglang_stage.sh` |
| vLLM 阶段及最终汇总 | `examples/sid-gr-inference/scripts/run_three_way_vllm_stage.sh` |
| 三方脚本公共函数 | `examples/sid-gr-inference/scripts/three_way_common.sh` |
| vLLM 离线 REC_BEAM runner | `examples/sid-gr-inference/tools/run_vllm_rec_beam_benchmark.py` |
| 结果归一化与汇总 | `examples/sid-gr-inference/tools/summarize_three_way_perf.py` |

除非命令明确切换目录，所有相对路径都从仓库根目录解析。

## 环境要求

SGLang 环境：

- `/opt/conda/bin/python`，Python 3.10；
- torch 2.9.1，CUDA 12.8；
- SID-GR editable 安装及已经验证的 kernel 依赖；
- SGLang `feature/beam_search` 源码；
- 本地 Qwen3-1.7B snapshot。

vLLM 环境：

- `/opt/conda/bin/python`，Python 3.10；
- torch 2.10.0，CUDA 12.8；
- 完整 CUDA Toolkit 12.8，且 `${CUDA_HOME}/bin/nvcc` 可用；
- `gcc`、`g++`、Git、AntCode CLI；
- 能访问 vLLM CMake 构建时拉取的依赖；
- 能看到与 SGLang 环境相同的模型、SGLang 源码、benchmark 代码和
  结果目录。

默认使用：

```bash
export PIP_INDEX_URL=https://artifacts.antgroup-inc.cn/simple/
export HF_ENDPOINT=https://hf-mirror.com
```

## 完整流程与实现说明

以下内容用于完整运行、修改脚本或排障。正常的 quick 流程不要一次性把本节
全部发给用户。

### 1. 运行 GR 和 SGLang

在 SGLang 环境中执行：

```bash
cd examples/sid-gr-inference

RUN_ID=three_way_l20 \
MODEL_DIR=/path/to/Qwen3-1.7B \
SGLANG_REPO=/path/on/persistent-disk/sglang_beam_search \
scripts/run_three_way_sglang_stage.sh
```

这个阶段会：

- 生成确定性的 token-ID JSONL workload；
- 记录 workload 的 SHA256；
- 运行 GR 和 SGLang 离线矩阵；
- 依次启动 GR、SGLang 服务并执行在线压测；
- 写入 `manifest.json`。

### 2. 原生编译 vLLM

切换到 vLLM 环境后执行：

```bash
cd examples/sid-gr-inference

PERSIST_ROOT=/path/on/persistent-disk \
scripts/bootstrap_three_way_vllm.sh
```

脚本会先检查 Python、torch、CUDA、nvcc 和目标提交。任一项不匹配就
直接退出，不修改 vLLM 安装。

核心编译命令是：

```bash
CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST=8.9 \
CMAKE_BUILD_TYPE=Release \
MAX_JOBS=8 \
NVCC_THREADS=2 \
FETCHCONTENT_BASE_DIR="${PERSIST_ROOT}/.cache/vllm-fetchcontent" \
/opt/conda/bin/python -m pip install \
  --no-deps --no-build-isolation -e "${VLLM_REPO}"
```

不要运行 `use_existing_torch.py`。它会改动源码；当前环境已经有匹配的
torch 2.10.0，不需要这一步。

### 3. 运行 vLLM 并汇总

```bash
RUN_ID=three_way_l20 \
PERSIST_ROOT=/path/on/persistent-disk \
MODEL_DIR=/path/to/Qwen3-1.7B \
SGLANG_REPO=/path/on/persistent-disk/sglang_beam_search \
scripts/run_three_way_vllm_stage.sh
```

运行前会检查：

- vLLM 和 SGLang 提交；
- 模型路径；
- workload checksum；
- vLLM 是否从 editable 源码加载；
- `vllm._C` 是否可导入；
- `AsyncLLM.rec_beam_search` 是否存在。

检查通过后，脚本运行 vLLM 离线和在线 benchmark，并生成三方汇总。

## 参数

所有 workload 参数都在 SGLang 阶段设置。vLLM 阶段从 manifest 读取，
不能静默换成另一组参数。

### 离线默认值

| 环境变量 | 默认值 |
| --- | --- |
| `CONTEXT_LENS` | `1000 5000` |
| `BEAM_WIDTHS` | `256` |
| `BATCH_SIZES` | `1 2 4 8` |
| `GR_DECODE_STEPS` | `2` |
| `BASELINE_DECODE_STEPS` | `3` |
| `WARMUP_RUNS` | `1` |
| `REPEAT` | `3` |

GR 的有效输出长度包含首个 token，因此使用 2 个 decode step；SGLang
和 vLLM 请求 3 个 token。三者最终都输出长度为 3 的候选。

### 在线默认值

| 环境变量 | 默认值 |
| --- | --- |
| `ONLINE_CONTEXT_LEN` | `5000` |
| `ONLINE_OUTPUT_LEN` | `3` |
| `ONLINE_BEAM_WIDTH` | `256` |
| `ONLINE_REQUESTS` | `64` |
| `ONLINE_MAX_CONCURRENCY` | `4` |
| `ONLINE_REQUEST_RATE` | `inf` |
| `ONLINE_WARMUP_REQUESTS` | `0` |
| `ONLINE_RANDOM_SEED` | `1` |
| `RUN_ONLINE` | `1` |

三次在线 benchmark 必须使用同样的输入长度、输出长度、请求数、并发、
request rate、warmup 数量和随机种子。

`--tokenize-prompt` 只能传给 `sglang` backend，不能传给 vLLM
backend。vLLM backend 会自动使用 `/v1/completions`，不要额外传
`--endpoint`。

目标机快速检查：

```bash
RUN_ID=three_way_quick QUICK=1 scripts/run_three_way_sglang_stage.sh
RUN_ID=three_way_quick scripts/run_three_way_vllm_stage.sh
```

## 结果目录和格式

结果位于：

```text
examples/sid-gr-inference/benchmark_artifacts/three_way/${RUN_ID}/
```

目录结构：

```text
manifest.json
workloads.sha256
workloads/*.jsonl
offline/{gr,sglang,vllm}_ctx*_beam*_req*.json
offline/normalized/{gr,sglang,vllm}_ctx*_beam*_req*.json
online/{gr,sglang,vllm}.jsonl
environment/*
logs/*
summary.md
summary.csv
summary.json
```

归一化离线结果使用 `runs[].outputs[].beams[]`。每个 beam 包含 rank、
生成 token IDs，以及框架能提供时的累计分数。

在线汇总只比较 request throughput、总时长和 E2E latency。不同框架对
beam candidate token 的计数方式不一致，不要直接比较它们报告的
output-token throughput。

## 本地检查

本地机器没有目标 GPU 环境时，只做静态检查：

```bash
bash -n examples/sid-gr-inference/scripts/three_way_common.sh \
  examples/sid-gr-inference/scripts/bootstrap_three_way_vllm.sh \
  examples/sid-gr-inference/scripts/run_three_way_sglang_stage.sh \
  examples/sid-gr-inference/scripts/run_three_way_vllm_stage.sh

git diff --check
```

没有在 L20 环境实际执行前，不得声称 CUDA 编译、模型加载或 benchmark
已经成功。

目标 vLLM 环境可用下面的命令确认实际加载路径：

```bash
/opt/conda/bin/python - <<'PY'
import vllm
import vllm._C
from vllm.v1.engine.async_llm import AsyncLLM

print(vllm.__file__)
print(vllm._C.__file__)
print(hasattr(AsyncLLM, "rec_beam_search"))
PY
```

## 常见问题

- 找不到 nvcc，或 CUDA 不是 12.8：停止，不回退到 wheel。
- vLLM 提交不对或工作区不干净：停止，不重置用户改动。
- workload checksum 失败：恢复或重跑 SGLang 阶段，不能在同一
  `RUN_ID` 下换 workload。
- vLLM 环境无法导入 `sglang.bench_serving`：检查
  `PYTHONPATH` 和客户端所需的 Python 包，不安装 SGLang runtime，
  不替换 torch。
- vLLM 容量校验失败：调整
  `VLLM_OFFLINE_MAX_NUM_SEQS`、
  `VLLM_OFFLINE_MAX_NUM_BATCHED_TOKENS` 或对应的在线容量变量，
  不改变 workload。
- 在线 JSONL 不完整：先检查对应 server log，再生成结论。

## 修改原则

- benchmark 相关修改只放在上面的入口文件或本 Skill 中。
- 保留已有 GR/SGLang benchmark 脚本和历史输出格式。
- 在线 workload 参数必须可调，并通过 manifest 在两个环境间同步。
- 不新增第二套在线压测客户端，继续使用 `sglang.bench_serving`。
- 未经用户明确要求，不扩展 accuracy、Nsight、多卡、镜像构建或部署。
