"""Articraft RL environment runtime schema.

All environment-owned state lives under ``state["rollout"]`` (a :class:`Rollout`).

Phase 2 refactoring: compile state extracted to :class:`CompileState`,
compaction tracking to :class:`CompactionState`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.workspace_docs import VirtualWorkspace

# 会修改 model.py 内容的工具名集合——用于 CompileState.mark_code_mutated() 触发
# edit_revision 递增，从而使 code_is_fresh() 返回 False，强制下次 compile_model 重新编译。
# "edit_lines" 是 v2 新增工具；在 v1 模式下不会被注册但存在于此集合中，零副作用
# （与 "apply_patch" 同理——它在 RL 环境中也未注册）。
MUTATING_TOOL_NAMES = frozenset({"apply_patch", "replace", "write_file", "edit_lines"})

SCHEMA_VERSION = "articraft-trajectory-v1"


@dataclass(frozen=True)
class Task:
    """Immutable view of one articraft task (one dataset row)."""

    record_id: str
    prompt_text: str
    category_slug: str | None = None
    sdk_package: str = "sdk"

    @classmethod
    def from_info(cls, info: dict[str, Any]) -> Task:
        return cls(
            record_id=info["record_id"],
            prompt_text=info["prompt_text"],
            category_slug=info.get("category_slug"),
            sdk_package=info.get("sdk_package", "sdk"),
        )


@dataclass
class TurnRecord:
    """One turn (one model response + tool execution results)."""

    turn: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    compile_attempted: bool = False
    compile_success: bool | None = None
    compile_signals: dict[str, Any] | None = None


@dataclass
class CompileState:
    """Compile feedback loop state — mirrors harness_compile.py CompileFeedbackLoop.

    Phase 1 fields were flat on Rollout; Phase 2 groups them here for clarity.
    """

    # Freshness tracking: edit_revision increments on write/replace,
    # last_revision updates on successful compile. Equal ⇒ code is "fresh".
    edit_revision: int = 0
    last_revision: int = -1

    # Compile result cache for freshness reuse / rubric scoring.
    last_bundle_dict: dict[str, Any] | None = None
    last_attempt_dict: dict[str, Any] | None = None

    # Failure streak (Phase 2 Feature #1).
    # Maps to CompileFeedbackLoop._last_compile_failure_sig / _consecutive_compile_failure_count.
    last_failure_sig: str | None = None
    consecutive_failure_count: int = 0

    # Termination nudge: env injects <compile_required> when model stops
    # without tool calls and code is stale. >3 nudges → force terminate.
    nudge_count: int = 0

    last_latency_ms: float | None = None

    def code_is_fresh(self) -> bool:
        return self.last_revision == self.edit_revision and self.last_revision >= 0

    def mark_code_mutated(self, tool_name: str) -> None:
        if tool_name not in MUTATING_TOOL_NAMES:
            return
        self.edit_revision += 1

    def mark_attempt(self, bundle: Any) -> None:
        self.last_attempt_dict = bundle.to_dict()

    def mark_success(self, bundle: Any) -> None:
        self.last_revision = self.edit_revision
        self.last_bundle_dict = bundle.to_dict()
        self.nudge_count = 0


@dataclass(frozen=True, slots=True)
class CompactionConfig:
    """Immutable compaction parameters — parsed from TOML ``[compaction]`` section.

    hard_threshold: prompt token ceiling before unconditional compaction fires.
      Derived as seq_len(16384) - system/tool_defs overhead (~2K) ≈ 14K.
    full_turns_to_keep: recent turns preserved verbatim during compaction,
      so the model always sees its latest compile errors + fix attempts.
    """

    hard_threshold: int = 14000
    full_turns_to_keep: int = 4

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> CompactionConfig:
        if not d:
            return cls()
        return cls(
            hard_threshold=d.get("hard_threshold", cls.hard_threshold),
            full_turns_to_keep=d.get("full_turns_to_keep", cls.full_turns_to_keep),
        )


@dataclass
class CompactionState:
    """Per-rollout compaction tracking for decide_compaction() parameters.

    These fields feed the cooldown logic in decide_compaction():
    - last_turn / last_prompt_tokens: "how many turns / tokens since last compaction?"
    - last_failure_sig: avoids re-triggering compaction for the same unresolved error.
    - count: observability only, does not affect decisions.
    """

    count: int = 0
    last_turn: int | None = None
    last_prompt_tokens: int | None = None  # post-compact value for growth_floor calc
    last_failure_sig: str | None = None


@dataclass
class Rollout:
    """Mutable runtime model of one articraft RL rollout.

    Compile state → :attr:`compile` (:class:`CompileState`).
    Compaction tracking → :attr:`compaction` (:class:`CompactionState`).
    """

    task: Task
    trajectory_id: str
    work_dir: Path
    max_turns: int
    script_path: Path
    virtual_workspace: VirtualWorkspace

    turns: list[TurnRecord] = field(default_factory=list)
    final_reward: float | None = None
    metadata: dict[str, object] | None = None

    compile: CompileState = field(default_factory=CompileState)
    compaction: CompactionState = field(default_factory=CompactionState)

    # Phase 2 Feature #2: one-shot edit_retry guidance injection.
    edit_retry_injected: bool = False

    @property
    def trajectory_short_id(self) -> str:
        return self.trajectory_id[:12]


def require_rollout(state: dict[str, Any]) -> Rollout:
    rollout = state.get("rollout")
    if not isinstance(rollout, Rollout):
        raise RuntimeError(
            "Articraft rollout state missing; setup_state likely failed"
        )
    return rollout
