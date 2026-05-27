# Session Handoff: Phase 2 Koala 提交与调试

## 前序 Session
- `.agents/session/2026-05-26-phase2-readiness-summary.md` — Phase 2 compaction 代码设计+审查完成，准备提交 Koala 测试
- `.agents/session/2026-05-25-articraft-kaola-full-debug.md` — Phase 1 Koala 调试全过程，最终训练成功启动

## 任务目的

将 Phase 2（context compaction + failure streak + edit_retry guidance）代码部署到 Koala 集群进行端到端验证，确认无 import 错误、API 兼容、OOM 消除后提交正式 8-GPU 训练。

## 执行内容

1. `aws s3 sync` 同步 prime-rl 代码（含未 commit 的 Phase 2 改动）到 S3
2. 尝试提交 debug pod → 跳板机连接失败（平台侧问题），改用 normal pod + `-c` 命令
3. 多次提交验证 pod，逐步修复 config 验证错误：
   - `orchestrator.eval.interval` 不能为 0（最小 1）
   - `max_inflight_rollouts` 与 `oversampling_factor * batch_size` 冲突
   - WANDB_API_KEY="notset" 导致 orchestrator 进程 AuthenticationError 崩溃
4. 发现 prime-rl 日志系统：orchestrator/inference/trainer 各有独立 `.log` 文件在 `output_dir/logs/`，koala logs 只显示 trainer stdout
5. 添加 `trap upload_logs EXIT` 上传日志到 S3，成功定位 wandb 认证问题
6. 修复：`--wandb.offline true --wandb.shared false` → 端到端验证通过
7. 创建 Qwen3.5-9B HF cache tar（19.3 GB），存于 `s3://arcwm-code-us-west-2/ericzyma/tools/hf_cache_qwen3.5-9b.tar`
8. 提交 8-GPU 正式训练 `ericzyma-job-normal-20260527-105409`，确认训练循环启动

## 调试经验

- **koala 跳板机连接**：debug pod 显示 Running 但 `koala ssh` 一直卡在"等待跳板机就绪"，无法 SSH。解决：改用 normal pod + `-c` 命令（不需要 SSH）
- **S3 FUSE append 静默失败**：`tee -a` 追加到已有文件在 FUSE 上不工作。解决：每次用唯一文件名（`verify-$(date +%H%M%S).log`）
- **koala logs 只有 trainer stdout**：orchestrator/inference 的 stderr 被重定向到 `$output_dir/logs/{orchestrator,inference,trainer}.log`。normal pod 挂了后文件随 `/local-ssd/` 丢失。解决：`trap` EXIT 时 `aws s3 sync` 上传日志
- **pydantic_config CLI override 顺序**：TOML 先被验证（计算 `max_inflight_rollouts=96`），CLI override 后再验证 → 冲突。解决：显式 override 所有相关字段
- **WANDB_MODE=disabled 对 prime-rl 无效**：prime-rl 内部子进程（orchestrator）自己调 `wandb.init()`，不受环境变量控制。必须用 `--wandb.offline true`
- **Qwen3.5-9B 无预缓存**：公共模型缓存（/asset/、/threed-code/public_models/）都没有此模型。已打包 tar 解决

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `scripts/phase2_verify.sh` | 全文 | 2-GPU 验证脚本（含 trap 上传日志、wandb offline） |
| `scripts/setup_kaola.sh` | L74, L112-140 | output guard check、后台 S3 sync |
| `scripts/envs/articraft.sh` | env_setup() | Articraft 环境初始化（code tar + dataset + deps） |
| `configs/articraft/rl_articraft_kaola.toml` | 全文 | Phase 2 训练配置（compaction@14K, dp6+train2） |
| `src/prime_rl/entrypoints/rl.py` | L170-280 | 日志文件重定向逻辑（orchestrator.log 等） |
| `src/prime_rl/configs/orchestrator.py` | L1096-1122 | resolve_batching 验证（max_inflight_rollouts 冲突源） |

## 当前运行状态

- **任务**: `ericzyma-job-normal-20260527-105409`（8 GPU，Phase 2 正式训练）
- **状态**: Training loop started (step 0)，正在做第一个 batch 的 rollout
- **EXP_NAME**: `articraft-phase2-dp6`
- **Wandb**: `articraft-rl` 项目，run `bda48650...`
- **S3 输出**: `s3://arcwm-code-us-west-2/ericzyma/experiments/articraft-phase2-dp6/output/`
- **HF cache tar**: ✅ 已创建 `s3://arcwm-code-us-west-2/ericzyma/tools/hf_cache_qwen3.5-9b.tar` (19.3 GB)

## 下一步任务

监控 Phase 2 训练：
1. 确认 Step 1 完成（reward > 0 表示环境正常工作）
2. 确认 `compaction_count > 0`（compaction 确实被触发）
3. 确认无 OOM（持续运行多个 step 无崩溃）
4. 根据 wandb 曲线评估训练质量，必要时调参
5. 训练稳定后 commit Phase 2 代码

## 初步方案

1. **查看 wandb metrics**：用 `wandb` Python API 或直接看面板 URL，关注 `reward/mean`、`compaction_count`（如果有）、`decode_len/mean`
2. **查看 koala logs**：`koala logs ericzyma-job-normal-20260527-105409` 确认 step 进度
3. **如果训练挂了**：下载 orchestrator.log（后台 sync 每 5 分钟同步到 `$OUTPUT_S3`），查看具体错误
4. **潜在风险**：
   - `[tool, user]` 消息序列经过 compaction 后是否触发 Qwen3 chat template 异常（verify 时 eval 跑了 10 个完整 rollout 都没问题，概率低）
   - 长时间训练可能因 HyperPod 节点驱逐重启（需 checkpoint 续训，ckpt interval=25 steps）
5. **训练稳定后**：commit 8 个改动文件（compaction.py、guidance.py、env.py、schema.py、rubric.py、artifact_manager.py、config、test）
