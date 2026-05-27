# Session Handoff: Articraft Phase 2 准备 — Token 分析与 OOM 定位

## 前序 Session
- `.agents/session/2026-05-25-articraft-kaola-full-debug.md` — Phase 1 完整调试记录（env_id 冲突、wandb 认证、训练启动成功）
- `.agents/session/2026-05-25-articraft-env-phase1-implementation.md` — Phase 1 代码实现

## 任务目的

在 Phase 1 训练成功启动后，分析 rollout 内容和 token 消耗，定位 OOM 根因，为 Phase 2 实施提供数据基础。

## 执行内容

- 从 S3 拉取 orchestrator.log 确认 Step 0 成功完成（reward=0.15 train, 0.31 eval）
- 确认 Trainer OOM 原因：seq_len=32768 时 backward pass 需 30.31 GiB，GPU 只剩 26.16 GiB
- 下载 eval_rollouts.jsonl，分析 10 个 eval rollout 的 token 分布
- 下载 trajectory.json（成功+失败各一个），分析 tool 调用序列和 compile 错误模式
- 将 4 个 commit push 到 fork `87003697/prime-rl` 的 `feat/articraft-env` 分支
- 合并 3 份 session handoff 为 `2026-05-25-articraft-kaola-full-debug.md`
- 更新 troubleshooting（3 条新记录）
- 更新 Phase 1/2 计划文件状态标记

## 调试经验

- **trajectory.json vs eval_rollouts.jsonl**：trajectory.json 只是摘要（tool names + compile signals），实际 token 内容在 `run_default/rollouts/step_0/eval_rollouts.jsonl` 中
- **trajectory_token_estimate 是累计推理消耗**：2M tokens 不是 rollout 序列长度（64K），而是多 turn 中重发历史的累计 input_tokens
- **模型输出短（60 tokens/turn）但 rollout 总长 64K**：token 主要来自 prompt（16K 固定）+ tool responses 累积 + tool call 参数（write_file 代码）

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `.agents/plans/articraft-env-integration_v2.md` | Phase 2 §2.2 Feature #6 | Context Window Management 方案（简单: token 估算截断 / 复杂: message 压缩） |
| `environments/articraft/articraft_env/env.py` | L110-155 | ArticraftEnv.__init__: prompt 构建、SDK docs 加载 |
| `environments/articraft/articraft_env/prompts.py` | 全文 | system_prompt + SDK docs 加载（占 prompt 16K 中的 11.6K） |
| `configs/articraft/rl_articraft_kaola.toml` | 全文 | 训练配置（当前 seq_len=32768, max_turns=50, max_completion_tokens=16384） |
| S3: `.../rollouts/step_0/eval_rollouts.jsonl` | 全文 | 10 个 eval rollout 完整数据（含 prompt/completion/metrics） |
| S3: `.../articraft-work/eval/` | meta.json + trajectory.json | rollout 摘要 + tool 调用序列 |

## 最终方案

Phase 1 已验证通过。Token 消耗分析结果：

| 组成 | Tokens | 占比 | 可优化 |
|------|--------|------|--------|
| Prompt: SDK docs | 11,600 | 18% | ✅ 截断/压缩 |
| Prompt: system + task | 4,400 | 7% | 部分可压缩 |
| Tool responses (50 turns) | 24,000 | 37% | ✅ 摘要旧 turns |
| Tool call args (write/replace) | 20,000 | 31% | ✅ 减少全量重写 |
| Assistant content | 3,200 | 5% | 已经很短 |
| **Total** | **~63,000** | | |

## 下一步任务

进行 Phase 2 内容构建，优先解决 token 预算溢出（Feature #6），然后按需启用其他 features。

## 初步方案

### 优先级 1: 解决 OOM（让训练能跑起来）

1. **短期 config 调参**（零代码改动，立即可提交）：
   - `max_turns: 50 → 15`（成功案例证明 10-15 turns 内有足够信号）
   - `seq_len: 32768 → 16384`（匹配 2 GPU 内存上限）
   - `max_completion_tokens: 16384 → 8192`（限制每 turn 输出）
   - 这组改动预计让 rollout 从 64K 降到 ~25K tokens，fit 进 16K seq_len

2. **Feature #6: Context Window Management**（~10-100 行改动）：
   - 简单方案：`env.py` 中估算累计 token，接近阈值时注入 `<context_budget_warning>` 或直接终止
   - 复杂方案：参考 `agent/providers/compaction_policy.py` 实现旧 message 压缩（保留最近 N 条完整 + 更早的只留 compile signals 摘要）

3. **Prompt 瘦身**（压缩 SDK docs）：
   - 当前 `load_sdk_docs_reference()` 加载完整 SDK 文档 11.6K tokens
   - 可截断到核心 API 列表 + 常用模式（~3-4K tokens）
   - 或用 BM25 动态选择相关文档片段（Phase 2 Feature #4 的思路）

### 优先级 2: 提升训练质量（观察调参后的训练曲线再决定）

4. **Feature #1: repeated/failure_streak**（~15 行）：检测模型重复同一错误，注入提示
5. **Feature #2: Guidance injection**（~60 行）：replace 失败时注入 `<edit_retry_guidance>`
6. **Feature #5: SFT warm-up**（如果从零 RL 完全不收敛）：合成单步成功 trajectory 做 warm-up

### 风险

- 减 max_turns 到 15 可能导致大部分 rollout 在 compile 之前就被截断（当前模型需要 3-5 turns 才做第一次 compile）
- SDK docs 截断可能让模型更频繁地幻觉 API（已经是主要失败模式）
- Message 压缩改变了训练数据分布，可能影响 TITO completion_mask 正确性
