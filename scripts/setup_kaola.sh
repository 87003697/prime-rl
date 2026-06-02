#!/bin/bash
# ============================================================================
# KOALA 环境恢复脚本 — 通用编排器
# ============================================================================
# 用法：
#   . scripts/setup_kaola.sh [--fast] [--resume] [--env blendergym]
#
# --fast   debug 模式（跳过数据集拷贝和 warmup）
# --resume 从已有 checkpoint 恢复训练（跳过 S3 output 存在检查）
# --env    环境插件名称（默认 blendergym），对应 scripts/envs/<name>.sh
#
# 注意：此脚本通过 source 执行（. scripts/setup_kaola.sh），set -euo pipefail
# 会影响调用方 shell。通过 koala submit -c 执行时无副作用（一次性 shell）；
# 在交互式 shell 中 source 时，后续命令也会受 set -e 约束。
#
# 环境变量（提交命令中 export）：
#   EXP_NAME      实验名称（必须设置，无默认值）
#   HF_MODEL      HuggingFace 模型全称（默认 Qwen/Qwen3.5-9B），用于 HF cache tar
#   HF_TOKEN      HuggingFace 认证
#   WANDB_API_KEY WandB 认证
# ============================================================================
set -euo pipefail

# --- 参数解析 ---
FAST_MODE=false
RESUME_MODE=false
ENV_NAME="blendergym"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast) FAST_MODE=true; shift ;;
        --resume) RESUME_MODE=true; shift ;;
        --env)
            if [[ $# -lt 2 ]]; then echo "ERROR: --env requires a name"; exit 1; fi
            ENV_NAME="$2"; shift 2 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$FAST_MODE" = true ]; then
    echo ">>> Fast mode: skip dataset copy & warmup"
fi

# --- 环境变量 ---
export HF_HOME="/local-ssd/hf_cache"
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set."
fi

# --- 路径配置 ---
if [ -z "${EXP_NAME:-}" ]; then
    echo "ERROR: EXP_NAME not set. Export it before running setup."
    echo "  e.g.: export EXP_NAME=articraft-0527-phase2-compaction"
    exit 1
fi
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ EXP_NAME 命名规范：{project}-{MMDD}-{description}                       │
# │                                                                         │
# │ 格式：                                                                   │
# │   {project}     — articraft / blendergym / ...                          │
# │   {MMDD}        — 提交日期（月日，4位）                                    │
# │   {description} — 2-3 个词描述本次实验重点                                 │
# │                                                                         │
# │ 示例：                                                                   │
# │   articraft-0525-phase1          Phase 1 baseline                       │
# │   articraft-0527-phase2-compaction  Phase 2 + context compaction        │
# │   articraft-0529-phase2-v2tools  Phase 2 + v2 tool set                  │
# │   blendergym-0513-9b-dp6        BlenderGym 9B dp6 训练                  │
# │                                                                         │
# │ 规则：                                                                   │
# │   - 全小写 + 连字符分隔                                                   │
# │   - 不含模型名（配置里已有）                                               │
# │   - 相同实验重跑用日期区分，不加 _2 后缀                                    │
# │   - S3 路径：experiments/{EXP_NAME}/output/                             │
# └─────────────────────────────────────────────────────────────────────────┘

HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-9B}"
HF_MODEL_SHORT=$(echo "${HF_MODEL}" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')

S3_PREFIX="/threed-code/ericzyma"
S3_EXP="${S3_PREFIX}/experiments/${EXP_NAME}"
OUTPUT_LOCAL="/local-ssd/prime-rl-output"
CKPT_LOCAL="/local-ssd/checkpoints/${EXP_NAME}"
CKPT_S3="${S3_EXP}/checkpoints"
OUTPUT_S3="${S3_EXP}/output"

# S3 API 路径（绕过 FUSE，用于写入）— FUSE 路径只用于读取/存在性检查
S3_BUCKET="s3://arcwm-code-us-west-2/ericzyma"
CKPT_S3_BUCKET="${S3_BUCKET}/experiments/${EXP_NAME}/checkpoints"
OUTPUT_S3_BUCKET="${S3_BUCKET}/experiments/${EXP_NAME}/output"
HF_CACHE_TAR="${S3_PREFIX}/tools/hf_cache_${HF_MODEL_SHORT}.tar"
PROJECT_DIR="/data/work/prime-rl"

if [ "$FAST_MODE" = false ] && [ "$RESUME_MODE" = false ] && [ -d "${OUTPUT_S3}/logs" ]; then
    echo "ERROR: S3 output already exists: ${OUTPUT_S3}/logs"
    echo "  Previous training data would be overwritten."
    echo "  To resume:      add --resume"
    echo "  To start fresh:  rclone purge threed-code:arcwm-code-us-west-2/${S3_EXP#/threed-code/}"
    echo "  Or use a different EXP_NAME."
    exit 1
fi

# ============================================================================
# 通用函数定义
# ============================================================================

# 将 S3 上预打包的 HuggingFace 模型缓存解压到本地 SSD。
# tar 文件路径由 HF_MODEL 派生：Qwen/Qwen3.5-9B → hf_cache_qwen3.5-9b.tar
setup_hf_cache() {
    echo "  HF model cache (${HF_MODEL})..."
    if [ ! -d "${HF_HOME}/hub" ]; then
        if [ -f "${HF_CACHE_TAR}" ]; then
            cat "${HF_CACHE_TAR}" | tar xf - -C /local-ssd
            echo "    Restored from ${HF_CACHE_TAR}"
        else
            echo "    No tar at ${HF_CACHE_TAR}, will download on first use"
        fi
    else
        echo "    Already present, skipping"
    fi
}

# 安装 prime-rl 主框架的 Python 依赖（含 flash-attn 加速库）。
setup_python_deps() {
    echo "  Python dependencies..."
    # --frozen: 严格按 uv.lock 安装，不重新解析 pyproject。
    # 之前用 --locked 会触发完整解析，被 7-day exclude-newer 窗口下
    # color-codeword/verifiers 元数据漂移坑过（lockfile 钉的版本看似不满足
    # 当前 PyPI 上元数据声明的依赖范围）。--frozen 直接信任锁文件。
    uv sync --frozen --extra flash-attn
}

# 启动后台 watchdog：每 30s 采样 cgroup memory 和 vLLM /liveness 端点。
# 这两个 log 由 EXIT trap 一同 sync 到 S3，下次 11h kill 至少留下内核侧痕迹
# 用于诊断（cgroup OOM 趋势 / vLLM hung 时刻）。
setup_watchdog() {
    local _mem_log="${OUTPUT_LOCAL}/logs/cgroup_mem.log"
    local _liveness_log="${OUTPUT_LOCAL}/logs/liveness.log"
    mkdir -p "$(dirname "${_mem_log}")"
    (while true; do
        local _ts="[$(date -u +%Y-%m-%dT%H:%M:%SZ)]"
        if [ -r /sys/fs/cgroup/memory.current ]; then
            echo "${_ts} mem=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo NA)" >> "${_mem_log}"
        fi
        # vLLM /liveness 端点（上游 a87badd6 引入），200 = healthy
        local _code
        _code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://localhost:8000/liveness 2>/dev/null || echo "ERR")
        echo "${_ts} liveness=${_code}" >> "${_liveness_log}"
        sleep 30
    done) &
    WATCHDOG_PID=$!
    echo "    Watchdog PID: ${WATCHDOG_PID} (cgroup mem + vLLM liveness, every 30s)"
}

# 启动后台进程，每 5 分钟将本地 SSD 上的训练产出同步到 S3（持久化）。
# shell EXIT 时触发最终同步，确保训练结束后不丢数据。
setup_s3_sync() {
    if [ "$FAST_MODE" = true ]; then
        echo "  Background sync: SKIPPED (fast mode)"
        return
    fi
    echo "  Starting background S3 sync..."
    setup_watchdog
    sync_all() {
        local _sync_log="${OUTPUT_LOCAL}/logs/s3_sync.log"
        mkdir -p "$(dirname "${_sync_log}")"
        # 使用 aws s3 sync 直接走 S3 API，绕过 FUSE 所有限制：
        #   - 不依赖 rename()、mtime、overwrite — 每次是 PUT 新对象
        #   - --delete: 清理远端多余文件（如旧 checkpoint）
        #   - --quiet: 不逐文件打印（减少日志噪音）
        if [ -d "${CKPT_LOCAL}" ]; then
            aws s3 sync "${CKPT_LOCAL}/" "${CKPT_S3_BUCKET}/" \
                --delete --quiet >> "${_sync_log}" 2>&1 || true
        fi
        if [ -d "${OUTPUT_LOCAL}" ]; then
            aws s3 sync "${OUTPUT_LOCAL}/" "${OUTPUT_S3_BUCKET}/" \
                --delete --exclude 'broadcasts/*' --exclude '*.bin' \
                --quiet >> "${_sync_log}" 2>&1 || true
        fi
    }
    (while true; do sleep 300; sync_all; done) &
    SYNC_PID=$!
    echo "    PID: ${SYNC_PID} (every 5 min, via S3 API)"
    echo "    ${CKPT_LOCAL} -> ${CKPT_S3_BUCKET} (--delete)"
    echo "    ${OUTPUT_LOCAL} -> ${OUTPUT_S3_BUCKET} (excl broadcasts/*.bin)"
    trap "kill ${SYNC_PID} 2>/dev/null || true; [ -n \"\${WATCHDOG_PID:-}\" ] && kill \"\${WATCHDOG_PID}\" 2>/dev/null || true; [ -n \"\${OPTIX_PID:-}\" ] && kill \"\${OPTIX_PID}\" 2>/dev/null || true; type _blendergym_cleanup &>/dev/null && _blendergym_cleanup; sync_all" EXIT
}

# ============================================================================
# 加载 env 插件 + 执行主流程
# ============================================================================
cd "${PROJECT_DIR}"

ENV_SCRIPT="${PROJECT_DIR}/scripts/envs/${ENV_NAME}.sh"
if [ ! -f "${ENV_SCRIPT}" ]; then
    echo "ERROR: env script not found: ${ENV_SCRIPT}"
    exit 1
fi
source "${ENV_SCRIPT}"

# --- 主流程 ---
# 顺序：python deps → env_setup（OPTIX 后台启动）→ hf_cache / s3_sync（与 OPTIX 并行）→ wait
# env_setup 依赖 python deps（uv pip install 需要 venv），所以必须在其后。
# hf_cache 与 env_setup 无依赖，放后面可以和 OPTIX warmup 并行执行。
echo "=== [1/7] Python dependencies ==="
setup_python_deps

echo "=== [2/7~5/7] Environment: ${ENV_NAME} ==="
env_setup

echo "=== [6/7] HF model cache ==="
setup_hf_cache

echo "=== [7/7] Background S3 sync ==="
setup_s3_sync

echo "=== Setup complete ==="
# prime-rl 在训练启动前执行 check_gpus_available()，检测到 GPU 上有进程会拒绝启动。
# OPTIX warmup 使用 GPU 0 编译 shader，必须等它完成后再训练。
# wait 期间 OPTIX 已与 [6/7] hf_cache + [7/7] s3_sync 并行执行了一部分。
if [ -n "${OPTIX_PID:-}" ]; then
    echo "  Waiting for OPTIX warmup (PID: ${OPTIX_PID})..."
    wait "${OPTIX_PID}" 2>/dev/null || true
    echo "  OPTIX warmup done."
fi
echo "=== Ready to train. ==="
