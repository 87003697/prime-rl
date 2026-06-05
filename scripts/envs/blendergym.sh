#!/bin/bash
# ============================================================================
# BlenderGym 环境插件 — 由 setup_kaola.sh 加载
# ============================================================================
# 实现 env_setup()，由 base 脚本在主流程中调用。
#
# 可用的 base 变量（source 时已定义）：
#   $S3_PREFIX_URI — S3 用户根 URI，如 s3://arcwm-code-us-west-2/ericzyma
#   $PROJECT_DIR   — prime-rl 代码目录
#   $OUTPUT_LOCAL  — 训练输出本地路径
#   $EXP_NAME     — 实验名称
#
# 约定：顶层只定义变量和函数。所有副作用（IO、安装）在 env_setup() 内执行。
# 函数命名用 setup_bg_ 前缀（BlenderGym）避免与 base 或其他 env 插件冲突。
# v1.4.0 适配（2026-06）：所有 tar 通过 s5cmd cat 从 S3 直接拉取，不再依赖
# /threed-code FUSE 挂载。
# ============================================================================

# --- BlenderGym 专属路径 ---
BLENDER_VERSION="4.2.0"
BLENDER_DIR="/local-ssd/blender-${BLENDER_VERSION}-linux-x64"
BLENDER_BIN="${BLENDER_DIR}/blender"
DATA_DIR="/local-ssd/blendergym"

BLENDER_TAR_URI="${S3_PREFIX_URI}/tools/blender-${BLENDER_VERSION}-linux-x64.tar"
BLENDERGYM_TAR_URI="${S3_PREFIX_URI}/data/blendergym.tar"
OPTIX_CACHE_TAR_URI="${S3_PREFIX_URI}/tools/optix_cache.tar"

setup_bg_install_system_libs() {
    echo "  [env] Installing system libraries (libegl1)..."
    apt-get update
    apt-get install -y --no-install-recommends libegl1
    rm -rf /var/lib/apt/lists/*
}

setup_bg_restore_blender() {
    echo "  [env] Restoring Blender ${BLENDER_VERSION}..."
    if [ ! -f "${BLENDER_BIN}" ]; then
        s5cmd cat "${BLENDER_TAR_URI}" | tar xf - -C /local-ssd
        echo "    Blender extracted"
    else
        echo "    Already present, skipping"
    fi
}

setup_bg_restore_dataset() {
    echo "  [env] Restoring dataset..."
    if [ ! -d "${DATA_DIR}/placement1" ]; then
        s5cmd cat "${BLENDERGYM_TAR_URI}" | tar xf - -C /local-ssd
        mv /local-ssd/bench_data "${DATA_DIR}"
        echo "    Restored from tar ($(du -sh "${DATA_DIR}" | cut -f1))"
    else
        echo "    Already present, skipping"
    fi
}

setup_bg_install_python_pkg() {
    echo "  [env] Installing blendergym package..."
    uv pip install --python /tmp/uv-venv/bin/python -e environments/blendergym
}

# OPTIX shader cache：恢复或编译 GPU shader，缓存到 /root/.nv/（~51MB）。
# cache 与 GPU 架构绑定（H200），换 GPU 型号需重新编译并更新 tar。
# 逻辑：已存在 → 跳过 | S3 有 tar → 恢复 | 都没有 → 编译 + 上传 S3
setup_bg_optix_warmup() {
    echo "  [env] OPTIX shader cache..."
    if [ -d "/root/.nv/ComputeCache" ]; then
        echo "    Already present, skipping"
        return
    fi
    if s5cmd ls "${OPTIX_CACHE_TAR_URI}" &>/dev/null; then
        s5cmd cat "${OPTIX_CACHE_TAR_URI}" | tar xf - -C /root
        echo "    Restored from S3 tar"
        return
    fi
    set -e
    echo "    No cache found, compiling from scratch (~6 min)..."
    uv run python -m blendergym.render \
        --blend "${DATA_DIR}/placement1/blender_file.blend" \
        --code "${DATA_DIR}/placement1/start.py" \
        --output-dir /local-ssd/warmup-render \
        --blender-bin "${BLENDER_BIN}" \
        --gpu 0 --resolution 64 --samples 1 \
        --compute-device OPTIX --timeout 600
    rm -rf /local-ssd/warmup-render
    # 先打包到本地，再用 s5cmd cp 上传（s5cmd 不支持 stdin → S3 stream）
    tar cf /tmp/optix_cache.tar -C /root .nv
    s5cmd cp /tmp/optix_cache.tar "${OPTIX_CACHE_TAR_URI}"
    rm -f /tmp/optix_cache.tar
    echo "    Compiled and saved to S3"
}

env_setup() {
    setup_bg_install_system_libs
    setup_bg_restore_blender

    echo "  [env] Blender: $(${BLENDER_BIN} --version | head -1)"

    setup_bg_restore_dataset
    setup_bg_install_python_pkg

    # OPTIX warmup must complete before starting Render Service to avoid
    # 6 Blender workers simultaneously JIT-compiling OPTIX kernels (OOM).
    setup_bg_optix_warmup

    LOG_DIR="/local-ssd/prime-rl-output/logs"
    mkdir -p "$LOG_DIR"

    export GPU_POOL="0,1,2,3,4,5"
    export BLENDER_BIN

    echo "  [env] Launching services..."
    if ! launcher_exports=$(uv run python -m blendergym.services.launcher --log-dir "$LOG_DIR"); then
        echo "  [env] ERROR: service launcher failed" >&2
        return 1
    fi
    eval "$launcher_exports"
    echo "  [env] Services launched (pids: ${SVC_PIDS:-})"

    _blendergym_cleanup() {
        echo "[env] Cleaning up services..."
        kill ${SVC_PIDS:-} 2>/dev/null || true
        wait 2>/dev/null || true
    }
}
