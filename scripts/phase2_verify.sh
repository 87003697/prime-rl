#!/bin/bash
# Phase 2 验证脚本 — Koala normal pod
# 策略：先跑 setup（它自带 set -euo pipefail），失败就失败，日志能看到哪步挂了。
# 关键改进：WANDB_MODE=disabled + 确保所有 env vars 有默认值

# 把所有输出写到 S3 FUSE（唯一文件名，避免 append 到已有对象的坑）
LOG_FILE="/threed-code/ericzyma/experiments/phase2-verify/verify-$(date +%H%M%S).log"
mkdir -p "$(dirname $LOG_FILE)"
exec > >(tee "$LOG_FILE") 2>&1

echo "=========================================="
echo "Phase 2 Verify Script Starting — $(date)"
echo "=========================================="
echo "Hostname: $(hostname)"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available yet"
echo ""

# 检查代码目录
if [ ! -d "/data/work/prime-rl" ]; then
    echo "FATAL: /data/work/prime-rl not found"
    ls -la /data/work/ 2>/dev/null
    exit 1
fi
cd /data/work/prime-rl
echo "Code dir: $(pwd)"
echo "Files: $(ls -1 | tr '\n' ' ')"
echo ""

# 预设环境变量（在 source setup_kaola.sh 之前，避免 set -u 炸掉）
export EXP_NAME="articraft-phase2-verify"
export HF_HOME="/local-ssd/hf_cache"
export HF_TOKEN="${HF_TOKEN:-notset}"
export WANDB_API_KEY="${WANDB_API_KEY:-notset}"
export WANDB_MODE="disabled"
export HF_HUB_DISABLE_XET=1

echo "ENV: EXP_NAME=$EXP_NAME HF_HOME=$HF_HOME WANDB_MODE=$WANDB_MODE"
echo ""

# ============================================================
echo "=== [1/4] Environment Setup ==="
echo "  Running: . scripts/setup_kaola.sh --fast --env articraft"
. scripts/setup_kaola.sh --fast --env articraft
echo "  [1/4] DONE"
echo ""

# ============================================================
echo "=== [2/4] Phase 2 Module Imports ==="
uv run python -c "
from articraft_env.compaction import decide_compaction, compact_messages, estimate_messages_tokens
from articraft_env.guidance import compile_signal_signature, maybe_inject_edit_code_guidance
from articraft_env.schema import CompileState, CompactionState
print('OK: Phase 2 imports (compaction, guidance, schema)')
"
echo "  [2/4] DONE"
echo ""

# ============================================================
echo "=== [3/4] Articraft API Compatibility ==="
uv run python -c "
from agent.feedback import render_compile_signals, compile_signal_bundle_from_exception
from agent.workspace_docs import load_sdk_docs_bundle
from pathlib import Path
import inspect

sig = inspect.signature(render_compile_signals)
assert 'repeated' in sig.parameters, 'FAIL: missing repeated param'
assert 'failure_streak' in sig.parameters, 'FAIL: missing failure_streak param'
print('OK: render_compile_signals(repeated=, failure_streak=)')

bundle = load_sdk_docs_bundle(Path('/data/work/articraft'), sdk_package='sdk')
txt = bundle.read_text('docs/sdk/references/quickstart.md')
assert len(txt) > 100, f'FAIL: quickstart too short: {len(txt)}'
print(f'OK: load_sdk_docs_bundle ({len(txt)} chars)')
"
echo "  [3/4] DONE"
echo ""

# ============================================================
echo "=== [4/4] Single Rollout Test ==="

# Trap: 无论成功/失败，都把日志上传 S3
upload_logs() {
    echo "--- Uploading logs to S3 ---"
    aws s3 sync /local-ssd/prime-rl-output/logs/ \
        s3://arcwm-code-us-west-2/ericzyma/experiments/phase2-verify/logs/ \
        --quiet 2>/dev/null || true
    echo "--- Logs uploaded ---"
}
trap upload_logs EXIT

# 覆盖 deployment：原配置需要 8 GPU (6 infer + 2 train)
# NCCL weight broadcast 要求至少 2 GPU，所以用 1 infer + 1 train
uv run rl @ configs/articraft/rl_articraft_kaola.toml \
    --max_steps 1 \
    --deployment.num_infer_gpus 1 \
    --deployment.num_train_gpus 1 \
    --orchestrator.batch_size 1 \
    --orchestrator.rollouts_per_example 1 \
    --orchestrator.oversampling_factor 1 \
    --orchestrator.max_inflight_rollouts 1 \
    --orchestrator.eval.interval 9999 \
    --orchestrator.train.env.0.num_workers 1 \
    --wandb.name "phase2-verify" \
    --wandb.offline true \
    --wandb.shared false \
    --ckpt.interval 9999
echo "  [4/4] DONE"
echo ""

echo "=========================================="
echo "=== ALL TESTS PASSED === $(date)"
echo "=========================================="
