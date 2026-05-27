# Session Handoff: Phase 2 Compaction 系统设计 + 代码审查

## 前序 Session
- `.agents/session/2026-05-26-articraft-phase2-token-analysis.md` — Token 消耗数据（64K rollout 拆解、OOM 根因定位）
- `.agents/session/2026-05-25-articraft-kaola-full-debug.md` — Phase 1 KAOLA 调试（env_id 冲突、wandb、训练启动成功）
- `.agents/session/2026-05-25-articraft-env-phase1-implementation.md` — Phase 1 代码实现

## 任务目的

设计 Phase 2 context compaction 系统方案（解决 64K rollout 导致的 OOM），经过多轮讨论迭代方案细节，最终用户自行实现代码后进行审查。

---

## 阶段 1：方案设计与多轮迭代

### Plan 文件创建

在 `.agents/plans/articraft-phase2-compaction-system.md` 中完成完整设计（~660 行），核心内容：

- **Feature #6 Context Window Management**：从 articraft 原版 `compaction_policy.py` 移植 4 级压力带 (55%/70%/85%/90%) + decide_compaction 决策逻辑
- **Feature #1 failure_streak**：compile 失败签名检测 + render_compile_signals(repeated=, failure_streak=)
- **Feature #2 edit_retry guidance**：replace 失败时注入 `<edit_retry_guidance>` 提示

### 多轮讨论要点

用户反复审视方案，提出 10+ 轮反馈，关键修改：

1. **命名对齐原版**："代码名字和内部原理应该尽可能跟 articraft 源代码对应"→ 重写所有命名对应表，不对应的给出原因
2. **Before/After 代码对比**："每一个代码改动都需要对比代码修改前后"→ 添加 4 个完整 Diff section（schema/env/compaction/config）
3. **添加详细注释**："代码改动里添加详细的注释，要不然我看不懂"→ 所有 Diff 加注释说明每行为什么这么写
4. **guidance.py 独立模块**：`maybe_inject_edit_code_guidance` 从 env.py inline 提取到独立 helper 文件
5. **Config 策略变更**：从"不要变更 config" → 最终同意 `seq_len=16384` 是训练侧安全网（与 compaction 配合）
6. **SDK docs 策略**：从 build_slim/minimal_sdk_docs → 改为只预加载 quickstart，其余通过 read_file 按需获取
7. **`get_prompt_messages` 架构**：从"自行重建 messages"修正为"先调 super()（完成 tool dispatch），再 compact 结果"
8. **无需 snapshot 字段**：删除 `compacted_messages` 快照——verifiers trajectory 自动传播压缩结果
9. **Schema 重构**：提取 CompileState + CompactionState dataclass，Rollout 精简

### 关键架构讨论

- **Extension property 与 compaction 的交互**：compaction 打断 extension property → interleave_rollout 产出多个 TrainingSample（不是 blocker，packer 处理）
- **seq_len 双重作用**：推理侧 max_seq_len 截断 prompt+completion；训练侧 packer 限制 micro-batch 上限
- **GRPO advantage 归因**：per-rollout（不是 per-turn），compaction 不改变这一行为。Per-turn credit assignment 留作 future work
- **规则压缩 vs LLM 摘要**：原版 trigger 后调 LLM 摘要；RL 版用确定性规则（把旧 turn 转成单行 "T3: compile_model: FAIL [isolated_part]"）避免额外推理开销
- **3.2 chars/token**：比 4 chars/token 更保守，避免低估 token 数

---

## 阶段 2：代码审查

用户自行实现全部代码（8 个文件，315 行净增）后请求审查：

- 逐文件 diff 对照 plan 命名对应表、伪代码、实施注意事项
- `ast.parse` 语法检查 7 个 Python 文件 ✅
- 确认 articraft 外部 API 兼容性：`render_compile_signals(repeated=, failure_streak=)` ✅、`DocsBundle.read_text()` ✅
- **审查结论：实现完整正确，与 plan 高度一致**

### 审查发现的小问题（不阻塞）

1. Eval env 无显式 compaction config（使用默认值，行为正确）
2. `summarize_turn_group` 未特殊跳过 read_file（多一行摘要不影响 budget）
3. `mark_compile_success` 只在无 failure 时调用（正确的行为变更，确保 failure 后不返回缓存）

---

## 调试经验

- **`get_prompt_messages` 不能跳过 super()**：基类内部调用 `env_response()` 执行 tool dispatch，如果 override 不调 super() 则 tools 完全不执行
- **compacted_messages 快照是多余的**：verifiers 的 trajectory 机制自动存储 `get_prompt_messages` 返回值，下一轮自动用压缩后版本作为 prev_prompt
- **多次 compaction 需要识别已有 `<compressed_history>` 块**：第一次 compact 后摘要被吸入 prefix，第二次必须检测并扩展而非创建新块
- **`compaction_last_prompt_tokens` 必须存 post-compact 值**：否则 cooldown 的 growth_floor 计算基于压缩前 token 数，永远满足增长条件
- **cached compile 路径不传 failure_streak**：模型未改代码重复调 compile 不算 failure streak（语义正确）

---

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `.agents/plans/articraft-phase2-compaction-system.md` | 全文 ~660 行 | 完整 Phase 2 设计方案（命名表 + 伪代码 + Diff + 注意事项） |
| `articraft_env/compaction.py` | 新文件 401 行 | decide_compaction（从原版移植）+ compact_messages + estimate |
| `articraft_env/guidance.py` | 新文件 53 行 | compile_signal_signature + maybe_inject_edit_code_guidance |
| `articraft_env/schema.py` | L53-110 | CompileState + CompactionState 提取 |
| `articraft_env/env.py` | L246-288 | get_prompt_messages override（compaction 入口） |
| `articraft_env/env.py` | L530-598 | _dispatch_compile failure streak 逻辑 |
| `articraft/agent/providers/compaction_policy.py` | 全文 | 原版参考（decide_compaction 源头） |
| `articraft/agent/harness_compile.py` | L119-123 | 原版 _compile_signal_signature |
| `articraft/agent/harness_guidance.py` | L243-275 | 原版 maybe_inject_edit_code_guidance |

## 下一步任务

1. S3 同步代码到 KAOLA（不先 commit，验证通过后再提交）
2. KAOLA debug pod 推理测试（单次 rollout 跑通，确认 API 兼容性）
3. 确认无问题后提交 8-GPU 正式训练，验证 OOM 消除
4. 全部通过后再 commit + push

## 初步方案

1. **S3 同步**：`s5cmd sync` 到 S3（含未 commit 的改动）
2. **KAOLA debug pod 推理测试**（1 GPU）：
   - 验证 import 链：articraft_env → compaction/guidance → agent.feedback/agent.workspace_docs
   - 验证 `render_compile_signals(bundle, repeated=True, failure_streak=3)` 不报 TypeError
   - 验证 `load_sdk_docs_bundle` + `DocsBundle.read_text("docs/sdk/references/quickstart.md")` 正常返回
   - 跑一个完整 rollout（确认 get_prompt_messages → compact_messages 路径无异常）
   - 确认 `[tool, user]` 序列经过 Qwen3 chat template 不报错
3. **KAOLA 8-GPU 正式训练**：
   - 提交 normal pod
   - 验证：无 OOM（Step 1+ 完成）、wandb reward ≥ 0.15、compaction_count > 0
4. **验证通过后 commit + push**
5. **风险**：
   - KAOLA 上 articraft 版本是否包含 `render_compile_signals` 的 `repeated`/`failure_streak` 参数
   - `load_sdk_docs_bundle` 路径依赖 articraft_root 正确
   - compact 后 `[tool, user]` 序列触发 Qwen3 chat template 异常
