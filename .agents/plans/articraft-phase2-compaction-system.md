# Plan: Articraft Phase 2 — Context Management + Feedback

## 设计哲学（对齐 v2 计划）

原 v2 计划的核心原则是 **"按训练曲线按需启用"**——不是一次性做完所有 feature，而是根据 Phase 1 训练观察逐步启用。

当前唯一的 **阻塞问题** 是 OOM (Feature #6)。其他 feature 是否启用取决于训练曲线：

| 观察到的问题 | 对应 Feature | 本次实施 | 理由 |
|-------------|-------------|---------|------|
| Context OOM | #6 context window mgmt | ✅ 必须 | 阻塞训练 |
| 模型反复无效 compile | #1 repeated/failure_streak | ✅ 顺带 | ~15 行，成本极低 |
| 模型 replace 死循环 | #2 edit_retry_guidance | ⚠️ 观察后决定 | 需要确认是否是真实问题 |
| QC 分数 plateau | #3 probe_model tool | ❌ 稍后 | 需要训练曲线数据 |
| 泛化能力差 | #4 find_examples tool | ❌ 稍后 | 需要训练曲线数据 |
| 从零 RL 完全失败 | #5 SFT warm-up | ❌ 稍后 | Phase 1 已有 0.15 reward，不算完全失败 |

---

## Feature #6: Context Window Management

### 两级方案（v2 计划原文）

> - 简单方案 (~10 行): 估算累计 token，接近阈值直接终止
> - 复杂方案 (~100 行): 参考 `agent/providers/compaction_policy.py` 实现 message 压缩

**本次实施复杂方案**（因为简单方案只是 early termination，浪费了已经产出的 rollout tokens）。

### 命名对应表

| RL 版 | Articraft 原版 | 对应 | 不对应原因 |
|--------|---------------|------|-----------|
| `SoftCompactionBand` | `SoftCompactionBand` | ✅ 同名同义 | — |
| `SOFT_COMPACTION_BANDS` | `SOFT_COMPACTION_BANDS` | ✅ 同名同值 | — |
| `HARD_PRESSURE_TRIGGER_RATIO` | `HARD_PRESSURE_TRIGGER_RATIO` | ✅ 同名同值 | — |
| `SOFT_COMPACTION_COOLDOWN_TURNS` | `SOFT_COMPACTION_COOLDOWN_TURNS` | ✅ 同名同值 | — |
| `SOFT_COMPACTION_GROWTH_FACTOR` | `SOFT_COMPACTION_GROWTH_FACTOR` | ✅ 同名同值 | — |
| `CompactionDecision` | `CompactionDecision` | ✅ 同名 | 删除 `cache_ratio` 字段 |
| `pressure_ratio()` | `pressure_ratio()` | ✅ 同名同逻辑 | — |
| `hard_pressure_trigger_tokens()` | `hard_pressure_trigger_tokens()` | ✅ 同名同逻辑 | — |
| `soft_compaction_band_for_pressure()` | `soft_compaction_band_for_pressure()` | ✅ 同名同逻辑 | — |
| `decide_compaction()` | `decide_compaction()` | ✅ 同名 | 删除 `cached_tokens` 参数 |
| 删除 `prompt_cache_ratio()` | `prompt_cache_ratio()` | ❌ 删除 | vLLM 不报告 cached_tokens（虽有 prefix caching 但无 API 级 usage 报告），无法计算 cache_ratio |
| 删除 `HIGH_CACHE_RATIO` | `HIGH_CACHE_RATIO = 0.60` | ❌ 删除 | 同上 |
| `compact_messages()` | 无（原版调 LLM） | ❌ 新增 | 原版 trigger 后调 LLM 摘要；RL 用规则压缩 |
| `summarize_turn()` | 无 | ❌ 新增 | 规则摘要的具体实现 |
| `estimate_messages_tokens()` | 无 | ❌ 新增 | 原版从 API response 取精确 usage；RL 在请求前需要估算 |
| `build_slim_sdk_docs()` | 无 | ❌ 不再需要 | 改为只预加载 quickstart（3.6K），其余 docs 模型用 read_file 按需读取 |
| `get_prompt_messages()` override | 无（原版无 verifiers） | ❌ 新增 | verifiers hook，原版直接操作 conversation list |

### 实现细节

#### `compaction.py` — 从原版 `compaction_policy.py` 复制 + 改造

**完全复制（不改）**：
- `SoftCompactionBand` dataclass
- `SOFT_COMPACTION_BANDS` tuple（3 个 band 定义）
- `HARD_PRESSURE_TRIGGER_RATIO = 0.90`
- `SOFT_COMPACTION_COOLDOWN_TURNS = 2`
- `SOFT_COMPACTION_GROWTH_FACTOR = 1.20`
- `CompactionDecision` dataclass（删 `cache_ratio` 字段）
- `pressure_ratio()` 函数
- `hard_pressure_trigger_tokens()` 函数
- `soft_compaction_band_for_pressure()` 函数

**复制后修改**：
- `decide_compaction()` — 删除所有 `cached_tokens` / `cache_ratio` 相关逻辑（3 处 if 分支）
  - 原因：vLLM 虽有 prefix caching（`enable_prefix_caching=true`），但不通过 API 向客户端报告 `cached_tokens`，无法计算 `cache_ratio`

**新增（原版无）**：
- `estimate_messages_tokens(messages)` — 3.2 chars/token 快速估算（偏保守，避免低估）
- `compact_messages(full_messages, ...)` — 核心压缩：根据 `decide_compaction()` 结果，将旧 turn 压缩为 `summarize_turn()` 输出
- `summarize_turn_group(turn_group, record, turn_index)` — 将一个 turn group (assistant + tool msgs) 压缩为单行摘要

**`compact_messages` 伪代码**（解决 turn 边界识别和多次 compaction 问题）：

```python
def compact_messages(
    messages: list[dict],
    turn_records: list[TurnRecord],
    full_turns_to_keep: int,
) -> list[dict]:
    """
    Turn 边界识别策略：每个 role="assistant" 开启一个新 turn。
    已压缩摘要标记：用 <compressed_history>...</compressed_history> XML 包裹。
    多次 compaction：识别已有 <compressed_history> 块，扩展它而非创建新块。
    """
    # 1. 找到第一个 assistant（对话起始点）
    first_assistant = next(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        len(messages),
    )

    # 2. 从 prefix 中分离 compressed_history（解决多次 compaction 重复块问题）
    existing_summary_lines: list[str] = []
    fixed_prefix: list[dict] = []
    for msg in messages[:first_assistant]:
        if (msg.get("role") == "user"
                and "<compressed_history>" in msg.get("content", "")):
            existing_summary_lines = _extract_summary_lines(msg["content"])
        else:
            fixed_prefix.append(msg)

    conversation = messages[first_assistant:]

    # 3. 按 assistant 开头分组为 turns
    turns: list[list[dict]] = []
    current: list[dict] = []
    for msg in conversation:
        if msg.get("role") == "assistant" and current:
            turns.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append(current)

    # 4. 分割：旧 turns（压缩）+ 新 turns（保留）
    compress_count = max(0, len(turns) - full_turns_to_keep)
    if compress_count == 0:
        return messages  # 无需压缩

    # 5. 压缩旧 turns 为摘要行（扩展已有摘要）
    for i in range(compress_count):
        turn_group = turns[i]
        turn_idx = len(existing_summary_lines)
        record = turn_records[turn_idx] if turn_idx < len(turn_records) else None
        line = summarize_turn_group(turn_group, record, turn_index=turn_idx)
        existing_summary_lines.append(line)

    # 6. 重建：prefix + 合并压缩块(一条 user msg) + 保留的 turns
    summary_content = "<compressed_history>\n" + "\n".join(existing_summary_lines) + "\n</compressed_history>"
    result = list(fixed_prefix)
    result.append({"role": "user", "content": summary_content})
    for turn_group in turns[compress_count:]:
        result.extend(turn_group)

    return result
```

#### `env.py` — 新增 `get_prompt_messages()` override

**原版无对应**（原版在 `harness.py` 主循环中直接操作 `conversation`）。

RL 版通过 verifiers hook 实现等效。

**⚠️ 关键架构约束**：基类 `get_prompt_messages()` 内部调用 `self.env_response()` 执行 tool dispatch。override 必须保留这个调用，否则 tools 不会执行。

**正确做法：先调基类（完成 tool dispatch），再 compact 结果**：

```python
# __init__ 中初始化（从 TOML compaction dict 读取）：
def __init__(self, ..., compaction: dict | None = None, **kwargs):
    cfg = compaction or {}
    self.compaction_hard_threshold: int = cfg.get("hard_threshold", 14000)
    self.full_turns_to_keep: int = cfg.get("full_turns_to_keep", 4)

# get_prompt_messages override：
async def get_prompt_messages(self, state) -> Messages:
    # 1. 调基类：prev_prompt + prev_completion + env_response（tool dispatch 在此发生）
    full_prompt = await super().get_prompt_messages(state)

    if len(state.get("trajectory", [])) == 0:
        return full_prompt  # Turn 0 无需压缩

    # 2. 在完整 prompt 上做 compaction 决策
    rollout = require_rollout(state)
    current_tokens = estimate_messages_tokens(full_prompt)
    decision = decide_compaction(
        prompt_tokens=current_tokens,
        hard_threshold=self.compaction_hard_threshold,
        consecutive_compile_failure_count=rollout.compile.consecutive_failure_count,
        last_compile_failure_sig=rollout.compile.last_failure_sig,
        last_soft_compaction_failure_sig=rollout.compaction.last_failure_sig,
        compactable_item_count=max(0, len(rollout.turns) - self.full_turns_to_keep),
        turn_number=len(rollout.turns),
        last_soft_compaction_turn_number=rollout.compaction.last_turn,
        last_soft_compaction_prompt_tokens=rollout.compaction.last_prompt_tokens,
    )

    # 3. 触发时就地 compact
    if decision.trigger:
        full_prompt = compact_messages(full_prompt, rollout.turns, self.full_turns_to_keep)
        rollout.compaction.count += 1
        rollout.compaction.last_turn = len(rollout.turns)
        rollout.compaction.last_prompt_tokens = estimate_messages_tokens(full_prompt)  # post-compact 值
        rollout.compaction.last_failure_sig = rollout.compile.last_failure_sig

    return full_prompt
```

**为什么不需要 `compacted_messages` 快照**：
- `get_prompt_messages()` 返回 compacted prompt 后，verifiers 存入 `trajectory[N]["prompt"]`
- 下一轮基类自动用 `trajectory[-1]["prompt"]`（已是压缩后版本）+ `trajectory[-1]["completion"]`
- compaction 结果通过 trajectory 机制自然传播，无需额外状态

#### `prompts.py` — 改为只预加载 quickstart

**原版预加载 3 个文档**（quickstart + probe-tooling + testing = 11.6K tokens）。

**RL 版改为只预加载 quickstart**（~3.6K tokens）。其余文档模型通过 `read_file(path="docs/sdk/references/testing.md")` 按需获取。

改动：修改 `load_sdk_docs_reference()` 的预加载列表，或在 `env.py` 中调用时只传 quickstart 路径。

```python
# 方案: 在 env.py __init__ 中替换 sdk_docs_context 的加载方式
# 原来: self.sdk_docs_context = load_sdk_docs_reference(...)  # 加载 3 个文档
# 改为: self.sdk_docs_context = load_sdk_docs_reference(..., preload_paths=("docs/sdk/references/quickstart.md",))
```

**不再需要 `build_slim_sdk_docs()` / `build_minimal_sdk_docs()`**——只有 3.6K 预加载，压力带中也不需要 SDK 截断逻辑。如果 3.6K 仍太大（CRITICAL 压力时），直接从 prompt 中删除 SDK 段落即可。

---

## Feature #1: `repeated`/`failure_streak`

### v2 计划原文

> ```
> schema.py:
>   Rollout:
> +   last_failure_sig: str | None = None         # harness_compile.py L54
> +   consecutive_failure_count: int = 0           # harness_compile.py L56
> 
> env.py:
>   _dispatch_compile():
> +   sig = hashlib.sha1(json.dumps(bundle.to_dict(), sort_keys=True).encode()).hexdigest()
> +   repeated = (sig == rollout.last_failure_sig) if has_failures else False
> +   # 传给 render_compile_signals(bundle, repeated=repeated, failure_streak=count)
> ```

### 命名对应表

| RL 版 | Articraft 原版 | 对应 |
|--------|---------------|------|
| `Rollout.last_failure_sig` | `CompileFeedbackLoop._last_compile_failure_sig` | ✅ 同义（简化命名） |
| `Rollout.consecutive_failure_count` | `CompileFeedbackLoop._consecutive_compile_failure_count` | ✅ 同义（简化命名） |
| `guidance.compile_signal_signature()` | `CompileFeedbackLoop._compile_signal_signature()` | ✅ 同逻辑，提取到 guidance.py |
| 直接调用 `render_compile_signals(repeated=, failure_streak=)` | `CompileFeedbackLoop._render_compile_tool_output()` | ✅ 去掉中间方法，直接传参 |

### 实现

~15 行改动，完全按 v2 计划描述，调用 `guidance.compile_signal_signature()`：

```python
# env.py _dispatch_compile() 中（try 和 except 两个路径都需要）:
from .guidance import compile_signal_signature

failures = [s for s in bundle.signals if s.severity == "failure"]
if failures:
    sig = compile_signal_signature(bundle.to_dict())
    repeated = sig == rollout.compile.last_failure_sig
    rollout.compile.last_failure_sig = sig
    rollout.compile.consecutive_failure_count += 1
else:
    repeated = False
    rollout.compile.last_failure_sig = None
    rollout.compile.consecutive_failure_count = 0

content = render_compile_signals(bundle, repeated=repeated,
                                 failure_streak=rollout.compile.consecutive_failure_count)
```

注：cached compile 路径（`code_is_fresh` 时）不传 `repeated`/`failure_streak` — 语义正确（模型未改代码重复调 compile 不算 failure streak）。

---

## Feature #2: Guidance Injection（按需启用）

### v2 计划原文

> | Guidance | 触发条件 | articraft 源码 | 注入内容 |
> |----------|---------|---------------|---------|
> | `edit_retry` | replace 失败 + "Could not find" | `harness_guidance.py` L243-275 | `<edit_retry_guidance>` XML |
> | `exact_geometry` | mutation 成功 + AST 发现 missing visual names | `harness_guidance.py` L143-179 | `<exact_geometry_contract>` XML |
> | `baseline_qc` | mutation 成功 + run_tests 含 compiler-owned QC | `harness_guidance.py` L181-215 | `<baseline_qc_guidance>` XML |
> 
> 所有 guidance 均为 **one-shot per rollout** + **user message 注入** + **TITO 兼容 (tool 在前 user 在后)**。

### 本次只实现 `edit_retry`（其余观察后决定）

**封装为独立模块 `guidance.py`**，对标原版 `harness_guidance.py` 的 `GuidanceInjector` 类。

原因：
- `exact_geometry` 需要 AST 分析 (`scan_code_contracts`)，依赖较重，且 Phase 1 数据未显示此为主要问题
- `baseline_qc` 需要 `scan_code_contracts`，同上
- `edit_retry` 最简单（检查 error 字符串即可），且 Phase 1 rollout 数据中确实有 replace 失败
- 独立模块方便后续扩展（加 exact_geometry / baseline_qc 时只改 guidance.py + env.py 调用处）

### 命名对应表

| RL 版 | Articraft 原版 | 对应 | 不对应原因 |
|--------|---------------|------|-----------|
| `maybe_inject_edit_code_guidance()` | `GuidanceInjector.maybe_inject_edit_code_guidance()` | ✅ 同名 | 方法 → 纯函数 |
| 无 `GuidanceInjector` class | `GuidanceInjector` class | ❌ 不使用 | 原版需维护 `_seen_*` 集合做 one-shot；RL 版可在 Rollout 上用 flag 替代 |
| `<edit_retry_guidance>` 文案 | `harness_guidance.py` L268-272 | ✅ 内容一致 | — |
| 无 `exact_geometry` | `_maybe_inject_exact_geometry_contract_guidance()` | ❌ 延后 | 需要 `scan_code_contracts()`，依赖重 |
| 无 `baseline_qc` | `_maybe_inject_baseline_qc_guidance()` | ❌ 延后 | 需要 `scan_code_contracts()`，同上 |
| 返回 `dict \| None` | `self._append_guidance_message(conversation, content)` | ❌ 形式不同 | verifiers 要求 env_response 返回 messages |
| one-shot 用 `Rollout.edit_retry_injected` flag | `self._seen_tool_error_sigs` set | ❌ 形式不同 | Rollout 是 per-rollout 状态，效果相同 |

### Articraft 原版 `edit_retry` 原文

```python
# harness_guidance.py L243-275
def maybe_inject_edit_code_guidance(self, conversation, *, tool_calls, tool_results):
    for tool_call, result in zip(tool_calls, tool_results, strict=False):
        func = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        func_name = func.get("name")
        if func_name != "replace":
            continue
        if not getattr(result, "error", None):
            continue
        if "Could not find the old_string in the code" not in result.error:
            continue
        sig = f"{func_name}_old_string_not_found"
        if sig in self._seen_tool_error_sigs:
            return
        self._seen_tool_error_sigs.add(sig)
        self._append_guidance_message(conversation, (
            "<edit_retry_guidance>\n"
            f"- Your last {func_name} failed because `old_string` did not match the file exactly.\n"
            '- Do NOT guess. Call `read_file(path="model.py")` again, then pick a smaller exact '
            'snippet from the current editable code as `old_string` and retry.\n'
            "- Keep edits surgical.\n"
            "</edit_retry_guidance>"
        ))
        return
```

### RL 版实现

**`guidance.py`（helper 模块）**：

```python
# environments/articraft/articraft_env/guidance.py

"""Guidance injection — 对标 harness_guidance.py 的 GuidanceInjector。

本次实现 edit_retry；后续按需扩展 exact_geometry / baseline_qc。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agent.models import CompileSignalBundle


# --- Feature #1: compile failure signature (对标 CompileFeedbackLoop._compile_signal_signature) ---

def compile_signal_signature(bundle_dict: dict[str, Any]) -> str:
    """SHA-1 of bundle dict. 同名同逻辑于 harness_compile.py L119-123."""
    sig_src = json.dumps(bundle_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(sig_src).hexdigest()


# --- Feature #2: edit_retry guidance (对标 GuidanceInjector.maybe_inject_edit_code_guidance) ---

EDIT_RETRY_CONTENT = (
    "<edit_retry_guidance>\n"
    "- Your last replace failed because `old_string` did not match the file exactly.\n"
    '- Do NOT guess. Call `read_file(path="model.py")` again, then pick a smaller exact '
    'snippet from the current editable code as `old_string` and retry.\n'
    "- Keep edits surgical.\n"
    "</edit_retry_guidance>"
)


def maybe_inject_edit_code_guidance(
    tool_name: str,
    tool_error: str | None,
    *,
    already_injected: bool,
) -> dict[str, Any] | None:
    """对标 GuidanceInjector.maybe_inject_edit_code_guidance()。
    
    差异:
    - 原版用 self._seen_tool_error_sigs set 做 one-shot
    - RL 版用 already_injected 参数（由调用方从 Rollout.edit_retry_injected 传入）
    - 原版 append 到 conversation；RL 版返回 dict 由 env_response append
    """
    if tool_name != "replace":
        return None
    if not tool_error:
        return None
    if "Could not find" not in tool_error and "old_string" not in tool_error.lower():
        return None
    if already_injected:
        return None
    return {"role": "user", "content": EDIT_RETRY_CONTENT}
```

**`env.py` env_response() 中调用**：

```python
# env.py env_response() 末尾，return 前
from .guidance import maybe_inject_edit_code_guidance

for tc, result in zip(tool_calls, tool_results):
    guidance_msg = maybe_inject_edit_code_guidance(
        tool_name=_tc_name(tc),
        tool_error=result.error,
        already_injected=rollout.edit_retry_injected,
    )
    if guidance_msg:
        rollout.edit_retry_injected = True
        result_messages.append(guidance_msg)
        break
```

---

## 文件改动汇总

| 文件 | 改动量 | 内容 |
|------|--------|------|
| `articraft_env/compaction.py` | ~200 行新建 | 从原版复制 decide_compaction 体系 + 新增 compact_messages/estimate/summarize |
| `articraft_env/guidance.py` | ~40 行新建 | `compile_signal_signature()` + `maybe_inject_edit_code_guidance()`（对标 `harness_guidance.py`） |
| `articraft_env/schema.py` | +15 行 | Rollout 新增字段（见下方完整列表） |
| `articraft_env/prompts.py` | ~5 行修改 | 预加载列表从 3 个文档改为只有 quickstart |
| `articraft_env/env.py` | +60 行 | `__init__` compaction 初始化 + `get_prompt_messages()` override + `_dispatch_compile` 调用 guidance + `env_response` 调用 guidance |
| `configs/articraft/rl_articraft_kaola.toml` | +3 行 | 添加 `[orchestrator.train.env.args.compaction]` 段（只加 hard_threshold） |
| **合计** | **~320 行** | |

注意：
- `guidance.py` 独立模块，后续加 `exact_geometry` / `baseline_qc` 只需扩展此文件
- Feature #2 只做 edit_retry 一种（不做 exact_geometry/baseline_qc）
- 不建独立 failure streak warning message（依赖 `render_compile_signals` 内置的文案）

---

## Config 变更

```toml
seq_len = 16384              # 32768 → 16384（训练侧安全网：限制 micro-batch 打包上限）

[orchestrator.train.env.args.compaction]
hard_threshold = 14000           # 对应原版 hard_threshold 概念（prompt tokens 上限）
full_turns_to_keep = 4           # 保留最近 4 个 turn 完整
```

**为什么同时需要 compaction + 减 seq_len**：
- `compaction (hard_threshold=14000)`: 控制推理侧——模型看到的 prompt 不超过 14K tokens
- `seq_len=16384`: 控制训练侧——packer 打包 micro-batch 不超过 16K tokens，避免 OOM
- 两者配合：compaction 保证每个 rollout sample ~14K，seq_len=16384 保证 packer 不会把两个 14K 打包成 28K（只能放一个 14K + padding 到 16K）
- 推理侧不受 seq_len=16384 影响：因为 compaction 后 prompt(14K) + completion(60) = 14K < 16K，不触发截断

**`seq_len` 的双重作用**（容易混淆的点）：
- **Rollout 时**：verifiers 用 `max_seq_len` 截断每个 step 的 `prompt_ids + completion_ids`，超出则 `is_truncated=True`
- **Training 时**：packer 用 `seq_len` 作为 micro-batch 打包上限，决定 GPU 内存峰值

---

## Schema 重构（评审建议采纳）

Phase 1 从 `CompileFeedbackLoop` 解构出的字段拍扁在 Rollout 顶层。Phase 2 新增 failure streak + compaction 字段后，字段膨胀——这是重新归并的好时机。

### 提取 `CompileState`

对标 articraft `CompileFeedbackLoop` 的状态和方法：

```python
@dataclass
class CompileState:
    """Compile feedback loop — 对标 harness_compile.py CompileFeedbackLoop。"""
    # Freshness tracking
    edit_revision: int = 0
    last_revision: int = -1                  # was: last_compile_revision
    last_bundle_dict: dict | None = None     # was: last_compile_bundle_dict
    last_attempt_dict: dict | None = None    # was: last_compile_attempt_dict

    # Failure streak (Phase 2 Feature #1)
    last_failure_sig: str | None = None      # 对应 _last_compile_failure_sig
    consecutive_failure_count: int = 0       # 对应 _consecutive_compile_failure_count

    # Termination nudge
    nudge_count: int = 0                     # was: compile_required_count

    # Observability
    last_latency_ms: float | None = None     # was: last_compile_latency_ms

    def code_is_fresh(self) -> bool:
        return self.last_revision == self.edit_revision and self.last_revision >= 0

    def mark_code_mutated(self, tool_name: str) -> None:
        if tool_name not in MUTATING_TOOL_NAMES:
            return
        self.edit_revision += 1

    def mark_attempt(self, bundle) -> None:
        self.last_attempt_dict = bundle.to_dict()

    def mark_success(self, bundle) -> None:
        self.last_revision = self.edit_revision
        self.last_bundle_dict = bundle.to_dict()
        self.nudge_count = 0
```

### 提取 `CompactionState`

对标 `decide_compaction()` 的 tracking 参数：

```python
@dataclass
class CompactionState:
    """Compaction tracking — maps to decide_compaction() params."""
    count: int = 0                           # 监控：触发了几次
    last_turn: int | None = None             # 对应 last_soft_compaction_turn_number
    last_prompt_tokens: int | None = None    # 对应 last_soft_compaction_prompt_tokens (post-compact)
    last_failure_sig: str | None = None      # 对应 last_soft_compaction_failure_sig
```

### 重构后的 Rollout

```python
@dataclass
class Rollout:
    # === 标准骨架 ===
    task: Task
    trajectory_id: str
    work_dir: Path
    max_turns: int
    turns: list[TurnRecord] = field(default_factory=list)
    final_reward: float | None = None       # ← 修复：rubric 中赋值（Phase 1 bug）
    metadata: dict[str, object] | None = None

    # === 执行上下文（setup 后不变）===
    script_path: Path
    virtual_workspace: VirtualWorkspace

    # === Compile 反馈循环 ===
    compile: CompileState = field(default_factory=CompileState)

    # === Phase 2: Guidance ===
    edit_retry_injected: bool = False

    # === Phase 2: Compaction ===
    compaction: CompactionState = field(default_factory=CompactionState)

    @property
    def trajectory_short_id(self) -> str:
        return self.trajectory_id[:12]
```

### env.py 调用变更对照

| Phase 1（现在） | Phase 2（重构后） |
|----------------|-----------------|
| `rollout.code_is_fresh()` | `rollout.compile.code_is_fresh()` |
| `rollout.mark_code_mutated(name)` | `rollout.compile.mark_code_mutated(name)` |
| `rollout.mark_compile_attempt(b)` | `rollout.compile.mark_attempt(b)` |
| `rollout.mark_compile_success(b)` | `rollout.compile.mark_success(b)` |
| `rollout.compile_required_count += 1` | `rollout.compile.nudge_count += 1` |
| `rollout.compile_required_count > 3` | `rollout.compile.nudge_count > 3` |
| `rollout.last_compile_bundle_dict` | `rollout.compile.last_bundle_dict` |
| `rollout.last_compile_attempt_dict` | `rollout.compile.last_attempt_dict` |
| `rollout.last_compile_latency_ms = x` | `rollout.compile.last_latency_ms = x` |
| `rollout.last_failure_sig` | `rollout.compile.last_failure_sig` |
| `rollout.consecutive_failure_count` | `rollout.compile.consecutive_failure_count` |

### 顺带修复：`final_reward` 未赋值（Phase 1 bug）

rubric.py 的 `score_rollout` 结尾补：
```python
rollout.final_reward = state["reward"]
```

注：不需要 `compacted_messages` 快照——verifiers 的 trajectory 机制自动传播 compaction 结果。

---

## 实施顺序

0. **TITO 兼容性验证**（硬性前置）：在 debug pod 上用 `tokenizer.apply_chat_template` 测试 `[tool, user]` 序列是否被 Qwen3 chat template 正确处理。如不兼容则 guidance 改为嵌入 tool message content。
1. **schema 重构**：提取 `CompileState` + `CompactionState`，重组 `Rollout`（纯重命名，不改逻辑）
2. **调用路径更新**：env.py / rubric.py / artifact_manager.py 的 `rollout.xxx` → `rollout.compile.xxx`；修复 `final_reward` bug
3. **验证 Phase 1 行为不变**（本地 import test 或简单 smoke test）
4. `compaction.py` — 从原版复制 `decide_compaction` + 新增 `compact_messages` / `estimate_messages_tokens`
5. `guidance.py` — `compile_signal_signature()` + `maybe_inject_edit_code_guidance()`
6. `env.py` 集成 — `__init__` compaction config + `get_prompt_messages` override + `_dispatch_compile` Feature #1 + `env_response` Feature #2
7. `prompts.py` — 预加载列表改为只有 quickstart
8. TOML config — 添加 `[orchestrator.train.env.args.compaction]`
9. **Token 估算校准**：用实际 eval_rollouts.jsonl 比较 3.2 chars/token vs tokenizer 精确编码，偏差 > 20% 则切换

Step 0 是写代码前的验证。Step 1-3 是无功能变更的重构，可先提交一个 commit。

---

## 验证

1. 用 S3 上的 `eval_rollouts.jsonl` 离线验证 compact_messages 压缩比
2. KAOLA 8-GPU 提交：无 OOM + Step 1+ 完成
3. 观察 wandb：reward ≥ 0.15，compaction 次数，truncation 率
4. 根据训练曲线决定是否启用 Feature #2 的 `exact_geometry` / `baseline_qc`

S3 数据: `s3://arcwm-code-us-west-2/ericzyma/experiments/articraft-9b-dp6/output/run_default/rollouts/step_0/eval_rollouts.jsonl`

---

## 评审反馈采纳汇总

| # | 反馈 | 采纳方案 |
|---|------|---------|
| 二 | Extension property 打断 → 训练样本碎片化 | 不是 blocker（interleave_rollout 已处理）。新增 `compaction_count` metric 监控碎片化程度 |
| 三 | Token 估算精度（4 chars/token 低估） | 改用 **3.2 chars/token**；实施后用实际 tokenizer 校准，如有大偏差再切换精确计算 |
| 四 | Message 重建不可靠 | **修正**：不维护快照，改用 `super().get_prompt_messages()` + 在结果上 compact。verifiers trajectory 自动传播压缩结果 |
| 五 | summarize_turn 信息损失 | 按 tool 类型差异化：compile 保留 severity + 前 2 个 signal 摘要；read_file 可完全删除；write/replace 只留 OK/FAIL |
| 六 | Compaction state 字段不完整 | **已采纳**：完整列出 6 个字段（见 Schema 字段列表） |
| 七 | SDK docs 缩减 distribution shift | 保持预加载 quickstart（与原版 3 个文档相比已缩减）。如果模型退化，Phase 2.1 恢复 testing.md |
| 八 | hard_threshold 推导 | `seq_len=16384` - system/tool_defs 固定开销 ~2K → **hard_threshold=14000**。保证 compaction 后 prompt fit 在 seq_len 内不被截断 |
| 九 | Guidance 注入 TITO 兼容性 | 实施前用 `tokenizer.apply_chat_template` 验证 `[tool, user]` 序列；如不兼容则将 guidance 嵌入最后一条 tool message content |

---

## 后续 Feature 启用条件（训练曲线观察后）

| Feature | 启用条件 | 依赖 |
|---------|---------|------|
| #2 exact_geometry + baseline_qc | 模型频繁因 visual name mismatch 或 baseline QC 重复失败 | 需实现 `scan_code_contracts()` |
| #3 probe_model tool | QC 分数 plateau，模型需要诊断工具 | 需 articraft `agent/tools/probe_model/` |
| #4 find_examples tool | 泛化差，模型不知道如何处理新 category | 需 BM25 index + example corpus |
| #5 | SFT warm-up | reward 持续 ≤ 0.1，模型完全不收敛 | 需合成数据脚本 |
| #6 Per-Turn Credit Assignment | Compaction 后某些 turn 被压缩掉但 advantage 仍平均分配，导致低质量 turn 获得与高质量 turn 相同的梯度信号 | 需修改 trainer advantage 计算 |

### Future Work: Per-Turn Credit Assignment

**现状**：GRPO advantage 是 per-rollout 的 — 同一 rollout 内所有 TrainingSample 共享同一个 advantage 值（基于 final reward 在 group 内的相对排名）。Compaction 不改变这一行为（与 Phase 1 完全一致）。

**问题**：当 rollout 有 20 个 turn，其中前 15 个 turn 在反复失败（compile error），最后 5 个 turn 修复了问题并获得高 reward。当前 GRPO 给所有 turn 相同的正 advantage，等于在强化前 15 个失败 turn 的行为。

**潜在方案**（按复杂度递增）：

1. **Per-turn reward shaping**：在 `rubric.py` 中给每个 turn 一个中间 reward（如 compile 成功 +0.1），advantage 按 turn 级别计算
2. **Advantage decay**：距离 final reward 越远的 turn，advantage 乘以衰减因子 γ^(T-t)
3. **Only-train-post-compaction samples**：被压缩掉的 turn 不参与训练（completion_mask 全 False），只训练保留的 turn
4. **Per-turn KL penalty**：对重复失败的 turn 增加 KL penalty，抑制反复尝试同一错误策略

**当前决策**：Phase 2 不实现。先观察 compaction + failure_streak guidance 能否间接缓解（通过提示模型避免重复错误）。如果训练曲线显示模型学到了「先反复失败再修复」的 pattern，则启用方案 2 或 3。

---

## 实施注意事项（评审精华）

1. **Cached compile 路径不传 `repeated`/`failure_streak`** — 这是正确的（code_is_fresh 时重复调 compile 不算 failure streak），无需修改
2. **Feature #1 需在 try 和 except 两个路径都插入** — exception 路径的 `compile_signal_bundle_from_exception()` 也会产出 `severity=="failure"` signal
3. **`compaction_last_prompt_tokens` 存 post-compact 值** — compact 后重新 `estimate_messages_tokens(full_prompt)`，使 cooldown 的 growth_floor 计算合理
4. **TITO 兼容性验证** — 实施前用 `tokenizer.apply_chat_template([tool_msg, user_guidance_msg])` 确认 Qwen3 模板正确处理此序列
5. **`compact_messages` 多次 compaction** — 用 `<compressed_history>` XML 标记包裹压缩块，后续 compaction 识别并扩展已有块（不创建新块）
6. **summarize_turn_group 按 tool 差异化** — compile: 保留 severity + 前 2 signal 摘要；read_file: 完全删除（模型可重新读）；write/replace: 只留 OK/FAIL
7. **turn-to-record 对齐检查** — `compact_messages` 中验证 message turns 数量 ≤ `len(turn_records) + 1`，mismatch 时 fallback 不压缩
8. **compaction 前检查 final_env_response** — 如 `state.get("final_env_response") is not None` 则跳过（即将终止，避免无效计算）
9. **`compressed_history` 用 `user` role** — 语义上略奇怪（不是用户说的话），训练初期观察模型是否被干扰；如有问题考虑嵌入 tool content 或 system role
10. **summarize_turn_group 多 tool call 格式** — 一个 turn 有多个 tool call 时合并为一行：`T3: replace(model.py) OK → compile_model: 2 failures [isolated_part, real_overlap]`


---

## 代码修改对比（Before → After）

### Diff 1: schema.py — Rollout 重构

**Before（Phase 1 现状，L51-113）：**

```python
@dataclass
class Rollout:
    task: Task
    trajectory_id: str
    work_dir: Path
    max_turns: int
    script_path: Path
    virtual_workspace: VirtualWorkspace
    turns: list[TurnRecord] = field(default_factory=list)
    final_reward: float | None = None
    metadata: dict[str, object] | None = None
    edit_revision: int = 0
    last_compile_revision: int = -1
    last_compile_bundle_dict: dict[str, Any] | None = None
    last_compile_attempt_dict: dict[str, Any] | None = None
    compile_required_count: int = 0
    last_compile_latency_ms: float | None = None

    def code_is_fresh(self) -> bool:
        return self.last_compile_revision == self.edit_revision and self.last_compile_revision >= 0

    def mark_code_mutated(self, tool_name: str) -> None:
        if tool_name not in MUTATING_TOOL_NAMES:
            return
        self.edit_revision += 1

    def mark_compile_attempt(self, bundle: Any) -> None:
        self.last_compile_attempt_dict = bundle.to_dict()

    def mark_compile_success(self, bundle: Any) -> None:
        self.last_compile_revision = self.edit_revision
        self.last_compile_bundle_dict = bundle.to_dict()
        self.compile_required_count = 0

    @property
    def trajectory_short_id(self) -> str:
        return self.trajectory_id[:12]
```

**After：**

```python
@dataclass
class CompileState:
    """将 Phase 1 散落在 Rollout 上的 compile 相关字段归并为一个子对象。
    
    对标 articraft 原版的 CompileFeedbackLoop 类（harness_compile.py L40-60）。
    原版是一个独立类，有 __init__ 和方法；这里用 dataclass 更轻量。
    """
    
    # ---- Freshness 追踪 ----
    # 判断"模型有没有改过代码但还没 compile"。
    # edit_revision 每次 write_file/replace 时 +1，
    # last_revision 在 compile 成功时设为 edit_revision 的值。
    # 两者相等 = 代码是"新鲜的"（已编译过，没改过）。
    edit_revision: int = 0                        # was: Rollout.edit_revision
    last_revision: int = -1                       # was: Rollout.last_compile_revision

    # ---- Compile 结果缓存 ----
    # last_bundle_dict: 最近一次"成功" compile 的完整结果（用于 freshness cache）
    # last_attempt_dict: 最近一次"任何" compile 的结果（用于 rubric 计算 reward）
    last_bundle_dict: dict[str, Any] | None = None   # was: Rollout.last_compile_bundle_dict
    last_attempt_dict: dict[str, Any] | None = None  # was: Rollout.last_compile_attempt_dict

    # ---- Phase 2 Feature #1: Failure Streak ----
    # 检测模型是否反复产出相同的 compile 错误。
    # last_failure_sig: 上一次失败的 SHA-1 签名（bundle dict 的 hash）
    # consecutive_failure_count: 连续相同签名的失败次数
    # 当签名变化（不同错误）时 count 重置为 1；成功时全部重置为 0。
    # 对标: CompileFeedbackLoop._last_compile_failure_sig / _consecutive_compile_failure_count
    last_failure_sig: str | None = None
    consecutive_failure_count: int = 0

    # ---- 终止催促 ----
    # 当模型没 tool_call 且代码未编译时，env 注入 <compile_required> 提醒。
    # 超过 3 次催促后强制终止 rollout。
    nudge_count: int = 0                          # was: Rollout.compile_required_count

    # ---- 可观测性 ----
    last_latency_ms: float | None = None          # was: Rollout.last_compile_latency_ms

    def code_is_fresh(self) -> bool:
        """代码是否已编译且此后未修改。对标 CompileFeedbackLoop.latest_code_is_fresh()。"""
        return self.last_revision == self.edit_revision and self.last_revision >= 0

    def mark_code_mutated(self, tool_name: str) -> None:
        """模型执行了 write_file/replace 等改代码的工具。对标 CompileFeedbackLoop.mark_code_mutated()。"""
        if tool_name not in MUTATING_TOOL_NAMES:
            return
        self.edit_revision += 1

    def mark_attempt(self, bundle: Any) -> None:
        """记录每次 compile 尝试（无论成功失败），rubric 读取此字段计算 reward。"""
        self.last_attempt_dict = bundle.to_dict()

    def mark_success(self, bundle: Any) -> None:
        """compile 无 blocking failure 时调用。重置 nudge 计数，更新 freshness。"""
        self.last_revision = self.edit_revision
        self.last_bundle_dict = bundle.to_dict()
        self.nudge_count = 0


@dataclass
class CompactionState:
    """Context compaction 的 per-rollout 追踪状态。
    
    这些字段对标 articraft compaction_policy.py 中 decide_compaction() 的参数：
    - last_turn / last_prompt_tokens: cooldown 逻辑需要（"上次压缩后过了几轮"、"token 增长了多少"）
    - last_failure_sig: 避免对同一错误重复触发压缩
    - count: 纯监控用，不影响决策
    """
    count: int = 0                              # 这个 rollout 触发了几次 compaction（监控）
    last_turn: int | None = None                # 上次 compaction 时是第几个 turn
    last_prompt_tokens: int | None = None       # 上次 compaction 后的 prompt token 数（post-compact）
    last_failure_sig: str | None = None         # 上次 compaction 时的 compile 失败签名


@dataclass
class Rollout:
    """重构后的 Rollout：compile 状态归并到 CompileState，compaction 状态归并到 CompactionState。
    
    顶层只保留"骨架"字段（task/turns/metadata）和执行句柄（script_path/workspace）。
    """
    # === 标准骨架（与 BlenderGym Rollout 对齐）===
    task: Task
    trajectory_id: str
    work_dir: Path
    max_turns: int
    script_path: Path                           # model.py 在磁盘上的路径
    virtual_workspace: VirtualWorkspace         # 虚拟文件系统（model.py + docs/）
    turns: list[TurnRecord] = field(default_factory=list)
    final_reward: float | None = None           # ← Phase 1 bug fix: rubric 中赋值
    metadata: dict[str, object] | None = None

    # === Compile 反馈循环（归并自 Phase 1 拍扁的 6 个字段 + Phase 2 新增 2 个）===
    compile: CompileState = field(default_factory=CompileState)

    # === Phase 2: Compaction 追踪 ===
    compaction: CompactionState = field(default_factory=CompactionState)

    # === Phase 2: Guidance 注入追踪 ===
    edit_retry_injected: bool = False            # one-shot: 每个 rollout 最多注入一次 edit_retry guidance

    @property
    def trajectory_short_id(self) -> str:
        return self.trajectory_id[:12]
```

---

### Diff 2: env.py __init__ — 添加 compaction 配置

**Before（L110-124）：**

```python
def __init__(
    self,
    articraft_root: str | Path = "/data/work/articraft",
    max_turns: int = 50,
    ...
    max_rollouts_per_example: int = 0,
    **kwargs: Any,
) -> None:
    # ... existing init (无 compaction) ...
```

**After：**

```python
def __init__(
    self,
    articraft_root: str | Path = "/data/work/articraft",
    max_turns: int = 50,
    ...
    max_rollouts_per_example: int = 0,
    compaction: dict[str, Any] | None = None,     # ← NEW
    **kwargs: Any,
) -> None:
    # ... existing init ...

    # Phase 2: compaction config
    cfg = compaction or {}
    self.compaction_hard_threshold: int = cfg.get("hard_threshold", 14000)
    self.full_turns_to_keep: int = cfg.get("full_turns_to_keep", 4)
```

---

### Diff 3: env.py _dispatch_compile — 添加 failure streak

**Before（L452-476）：**

```python
async def _dispatch_compile(self, rollout: Rollout) -> ToolResult:
    if rollout.code_is_fresh() and rollout.last_compile_bundle_dict is not None:
        # ... cached path ...
        return ToolResult(output=cached_text)

    t0 = time.monotonic()
    try:
        report = await asyncio.to_thread(compile_urdf_report_maybe_timeout, ...)
        rollout.last_compile_latency_ms = (time.monotonic() - t0) * 1000
        bundle = report.signal_bundle
        content = render_compile_signals(bundle)
        rollout.mark_compile_attempt(bundle)
        rollout.mark_compile_success(bundle)
        return ToolResult(output=content)

    except Exception as exc:
        rollout.last_compile_latency_ms = (time.monotonic() - t0) * 1000
        bundle = compile_signal_bundle_from_exception(exc)
        content = render_compile_signals(bundle)
        rollout.mark_compile_attempt(bundle)
        return ToolResult(output=content, error=str(exc))
```

**After：**

```python
async def _dispatch_compile(self, rollout: Rollout) -> ToolResult:
    # 导入 guidance.py 中的签名计算函数（对标 CompileFeedbackLoop._compile_signal_signature）
    from .guidance import compile_signal_signature

    # ---- Freshness Cache 路径（不变）----
    # 如果模型没改代码就重复调 compile，直接返回缓存的上次结果。
    # 注意：这里不传 repeated/failure_streak，因为重复调 compile 不算 failure streak。
    if rollout.compile.code_is_fresh() and rollout.compile.last_bundle_dict is not None:
        # ... cached path (unchanged, 不传 repeated/failure_streak) ...
        return ToolResult(output=cached_text)

    t0 = time.monotonic()
    try:
        # 执行真正的编译（在线程池中运行，避免阻塞 async loop）
        report = await asyncio.to_thread(compile_urdf_report_maybe_timeout, ...)
        rollout.compile.last_latency_ms = (time.monotonic() - t0) * 1000
        bundle = report.signal_bundle

        # ============ Feature #1: Failure Streak Tracking (try path) ============
        # 目的：检测模型是否反复产出相同错误，如果是则在 compile output 中追加警告文案。
        # 
        # 逻辑对标 harness_compile.py L125-140 _render_compile_tool_output():
        #   1. 提取所有 severity="failure" 的 signal
        #   2. 如果有失败：计算整个 bundle 的 SHA-1 签名
        #   3. 与上次失败签名比较 → repeated=True/False
        #   4. 更新签名 + 累加连续失败次数
        #   5. 如果没有失败（compile 成功）：重置签名和计数
        failures = [s for s in bundle.signals if s.severity == "failure"]
        if failures:
            # 计算这次失败的"指纹"——如果和上次一样，说明模型没修好
            sig = compile_signal_signature(bundle.to_dict())
            repeated = sig == rollout.compile.last_failure_sig
            rollout.compile.last_failure_sig = sig
            rollout.compile.consecutive_failure_count += 1
        else:
            # compile 成功：重置 streak（模型修好了）
            repeated = False
            rollout.compile.last_failure_sig = None
            rollout.compile.consecutive_failure_count = 0

        # 传入 repeated 和 failure_streak 后，render_compile_signals 会自动：
        # - repeated=True 时追加 "This failure matches the previous compile attempt."
        # - failure_streak>=3 时追加 "This is compile failure N in a row."
        # - 根据错误类型生成 response_rules (suggested next steps)
        content = render_compile_signals(bundle, repeated=repeated,
                                         failure_streak=rollout.compile.consecutive_failure_count)
        # ============ End Feature #1 ============

        rollout.compile.mark_attempt(bundle)
        if not failures:
            rollout.compile.mark_success(bundle)
        return ToolResult(output=content)

    except Exception as exc:
        rollout.compile.last_latency_ms = (time.monotonic() - t0) * 1000
        # compile_signal_bundle_from_exception 会产出一个 severity="failure" 的 signal
        bundle = compile_signal_bundle_from_exception(exc)

        # ============ Feature #1: Failure Streak (except path) ============
        # 与 try path 相同的逻辑——exception 也算一次 compile 失败
        sig = compile_signal_signature(bundle.to_dict())
        repeated = sig == rollout.compile.last_failure_sig
        rollout.compile.last_failure_sig = sig
        rollout.compile.consecutive_failure_count += 1
        # ============ End Feature #1 ============

        content = render_compile_signals(bundle, repeated=repeated,
                                         failure_streak=rollout.compile.consecutive_failure_count)
        rollout.compile.mark_attempt(bundle)
        return ToolResult(output=content, error=str(exc))
```

---

### Diff 4: env.py env_response — 添加 guidance 注入

**Before（L328-339，return 前）：**

```python
    # -- turn record --
    has_compile = "compile_model" in tc_names
    turn = TurnRecord(
        turn=len(rollout.turns),
        tool_calls=[{"name": n} for n in tc_names],
        compile_attempted=has_compile,
        compile_success=rollout.code_is_fresh() if has_compile else None,
        compile_signals=rollout.last_compile_attempt_dict,
    )
    rollout.turns.append(turn)
    return result_messages
```

**After：**

```python
    # ============ Feature #2: Edit Retry Guidance ============
    # 目的：当模型的 replace 操作因 "old_string 找不到" 失败时，
    # 注入一条 user message 引导模型先 read_file 再重试。
    #
    # 对标: harness_guidance.py L243-275 GuidanceInjector.maybe_inject_edit_code_guidance()
    # 差异: 原版用 GuidanceInjector 类 + _seen_tool_error_sigs set 做 one-shot；
    #       RL 版用 Rollout.edit_retry_injected bool flag（效果相同：每个 rollout 最多注入一次）。
    #
    # 为什么是 user message: TITO 兼容——user role 的 completion_mask=False，trainer 不会对它计算梯度。
    from .guidance import maybe_inject_edit_code_guidance

    for tc, result in zip(tool_calls, tool_results):
        guidance_msg = maybe_inject_edit_code_guidance(
            tool_name=_tc_name(tc),        # 只对 "replace" 工具触发
            tool_error=result.error,        # 检查是否包含 "Could not find"
            already_injected=rollout.edit_retry_injected,  # one-shot 控制
        )
        if guidance_msg:
            rollout.edit_retry_injected = True   # 标记：这个 rollout 已注入过，不再重复
            result_messages.append(guidance_msg)  # 追加到 env_response 返回的 messages 中
            break  # 每个 turn 最多注入一条 guidance
    # ============ End Feature #2 ============

    # -- turn record（注意：调用路径从 rollout.xxx 改为 rollout.compile.xxx）--
    has_compile = "compile_model" in tc_names
    turn = TurnRecord(
        turn=len(rollout.turns),
        tool_calls=[{"name": n} for n in tc_names],
        compile_attempted=has_compile,
        compile_success=rollout.compile.code_is_fresh() if has_compile else None,  # was: rollout.code_is_fresh()
        compile_signals=rollout.compile.last_attempt_dict,  # was: rollout.last_compile_attempt_dict
    )
    rollout.turns.append(turn)
    return result_messages
```

---

### Diff 5: env.py — 新增 get_prompt_messages override

**Before：** 不存在（使用基类默认实现）

**After：** 见上方 "Feature #6 实现细节 → env.py" 章节。

---

### Diff 6: prompts.py / env.py — SDK docs 预加载缩减

**Before（env.py L149-151）：**

```python
self.sdk_docs_context: str = load_sdk_docs_reference(
    self.articraft_root, sdk_package=self.sdk_package,
)
# load_sdk_docs_reference 预加载 quickstart + probe-tooling + testing (11.6K tokens)
```

**After：**

```python
self.sdk_docs_context: str = load_sdk_docs_reference(
    self.articraft_root,
    sdk_package=self.sdk_package,
    preload_paths=("docs/sdk/references/quickstart.md",),  # 只预加载 quickstart (~3.6K)
)
```

需同步修改 `load_sdk_docs_reference()` 或 `workspace_docs.py` 接受 `preload_paths` 参数。

---

### Diff 7: rl_articraft_kaola.toml — seq_len + compaction 配置

**Before：**

```toml
seq_len = 32768
# ... 无 compaction 段 ...
```

**After：**

```toml
# 训练侧安全网：限制 micro-batch 打包上限，防止 OOM。
# Compaction 保证每个 rollout sample ~14K tokens，seq_len=16384 保证
# packer 不会把两个 14K 打包成 28K（只能放一个 14K + padding）。
# 推理侧不受影响：compaction 后 prompt(14K) + completion(60) = 14K < 16K，不截断。
seq_len = 16384

[orchestrator.train.env.args.compaction]
# hard_threshold: prompt token 上限。对应原版 compaction_policy.py 的 hard_threshold 概念。
# 推导: seq_len(16384) - system/tool_defs 固定开销(~2K) = ~14K 可用于 prompt。
hard_threshold = 14000
# full_turns_to_keep: compaction 时保留最近多少个 turn 不压缩。
# 保留 4 个 turn 让模型看到最近的 compile 错误 + 修复尝试。
full_turns_to_keep = 4
```

---

### Diff 8: rubric.py — 修复 final_reward 未赋值 bug

**Before（score_rollout 结尾，无此行）：**

```python
# rollout.final_reward 从未被设置，artifact meta.json 永远写 null
```

**After：**

```python
rollout.final_reward = state["reward"]
```
