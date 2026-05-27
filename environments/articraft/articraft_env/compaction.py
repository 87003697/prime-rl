"""Context window management — ported from agent/providers/compaction_policy.py.

Copied verbatim:
  SoftCompactionBand, SOFT_COMPACTION_BANDS, HARD_PRESSURE_TRIGGER_RATIO,
  SOFT_COMPACTION_COOLDOWN_TURNS, SOFT_COMPACTION_GROWTH_FACTOR,
  CompactionDecision, pressure_ratio, hard_pressure_trigger_tokens,
  soft_compaction_band_for_pressure

Modified:
  decide_compaction — removed cached_tokens / cache_ratio logic
                      (vLLM does not expose cached_tokens via API)

New:
  estimate_messages_tokens — 3.2 chars/token heuristic
  compact_messages — rule-based history compression
  summarize_turn_group — per-turn summary generation
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .schema import TurnRecord

# ---------------------------------------------------------------------------
# Copied from compaction_policy.py (unchanged)
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 3.2

@dataclass(frozen=True, slots=True)
class SoftCompactionBand:
    min_pressure_ratio: float
    min_failure_streak: int
    min_compactable_items: int
    name: str


SOFT_COMPACTION_BANDS: tuple[SoftCompactionBand, ...] = (
    SoftCompactionBand(
        min_pressure_ratio=0.85,
        min_failure_streak=3,
        min_compactable_items=2,
        name="high_pressure",
    ),
    SoftCompactionBand(
        min_pressure_ratio=0.70,
        min_failure_streak=4,
        min_compactable_items=2,
        name="medium_pressure",
    ),
    SoftCompactionBand(
        min_pressure_ratio=0.55,
        min_failure_streak=5,
        min_compactable_items=3,
        name="early_pressure",
    ),
)

HARD_PRESSURE_TRIGGER_RATIO = 0.90
SOFT_COMPACTION_COOLDOWN_TURNS = 2
SOFT_COMPACTION_GROWTH_FACTOR = 1.20


@dataclass(frozen=True, slots=True)
class CompactionDecision:
    trigger: str | None
    reason: str
    pressure_ratio: float | None = None
    soft_failure_threshold: int | None = None
    min_compactable_items: int | None = None
    hard_trigger_tokens: int | None = None


def hard_pressure_trigger_tokens(hard_threshold: int | None) -> int | None:
    if not isinstance(hard_threshold, int) or hard_threshold <= 0:
        return None
    return max(1, math.ceil(hard_threshold * HARD_PRESSURE_TRIGGER_RATIO))


def pressure_ratio(*, prompt_tokens: int | None, hard_threshold: int | None) -> float | None:
    if (
        not isinstance(prompt_tokens, int)
        or prompt_tokens < 0
        or not isinstance(hard_threshold, int)
        or hard_threshold <= 0
    ):
        return None
    return prompt_tokens / hard_threshold


def soft_compaction_band_for_pressure(value: float | None) -> SoftCompactionBand | None:
    if value is None:
        return None
    for band in SOFT_COMPACTION_BANDS:
        if value >= band.min_pressure_ratio:
            return band
    return None


# ---------------------------------------------------------------------------
# decide_compaction — modified: removed cached_tokens / cache_ratio branches
# ---------------------------------------------------------------------------

def decide_compaction(
    *,
    prompt_tokens: int | None,
    hard_threshold: int | None,
    consecutive_compile_failure_count: int,
    last_compile_failure_sig: str | None,
    last_soft_compaction_failure_sig: str | None,
    compactable_item_count: int,
    turn_number: int,
    last_soft_compaction_turn_number: int | None,
    last_soft_compaction_prompt_tokens: int | None,
) -> CompactionDecision:
    """Decide whether to trigger context compaction.

    Two trigger paths:
    - "hard_pressure": prompt_tokens >= 90% of hard_threshold → unconditional safety valve.
    - "compile_plateau": model stuck in repeated compile failures + context is large enough
      that summarizing old turns would reclaim meaningful space.

    Compared to original (compaction_policy.py): removed all cached_tokens / cache_ratio
    logic because vLLM does not report cached_tokens via API (despite having prefix caching).
    """
    current_pressure = pressure_ratio(
        prompt_tokens=prompt_tokens,
        hard_threshold=hard_threshold,
    )
    hard_trigger = hard_pressure_trigger_tokens(hard_threshold)

    if (
        hard_trigger is not None
        and isinstance(prompt_tokens, int)
        and prompt_tokens >= hard_trigger
    ):
        return CompactionDecision(
            trigger="hard_pressure",
            reason="hard_pressure_band",
            pressure_ratio=current_pressure,
            hard_trigger_tokens=hard_trigger,
        )

    if consecutive_compile_failure_count <= 0 or not last_compile_failure_sig:
        return CompactionDecision(
            trigger=None,
            reason="no_compile_plateau",
            pressure_ratio=current_pressure,
            hard_trigger_tokens=hard_trigger,
        )

    if last_compile_failure_sig == last_soft_compaction_failure_sig:
        return CompactionDecision(
            trigger=None,
            reason="signature_already_soft_compacted",
            pressure_ratio=current_pressure,
            hard_trigger_tokens=hard_trigger,
        )

    band = soft_compaction_band_for_pressure(current_pressure)
    if band is None:
        return CompactionDecision(
            trigger=None,
            reason="soft_pressure_too_low",
            pressure_ratio=current_pressure,
            hard_trigger_tokens=hard_trigger,
        )

    required_failure_streak = band.min_failure_streak

    if consecutive_compile_failure_count < required_failure_streak:
        return CompactionDecision(
            trigger=None,
            reason="compile_plateau_below_threshold",
            pressure_ratio=current_pressure,
            soft_failure_threshold=required_failure_streak,
            min_compactable_items=band.min_compactable_items,
            hard_trigger_tokens=hard_trigger,
        )

    if compactable_item_count < band.min_compactable_items:
        return CompactionDecision(
            trigger=None,
            reason="insufficient_compactable_history",
            pressure_ratio=current_pressure,
            soft_failure_threshold=required_failure_streak,
            min_compactable_items=band.min_compactable_items,
            hard_trigger_tokens=hard_trigger,
        )

    if last_soft_compaction_turn_number is not None:
        turns_since = turn_number - last_soft_compaction_turn_number
        if turns_since < SOFT_COMPACTION_COOLDOWN_TURNS:
            growth_floor: int | None = None
            if isinstance(last_soft_compaction_prompt_tokens, int):
                growth_floor = math.ceil(
                    last_soft_compaction_prompt_tokens * SOFT_COMPACTION_GROWTH_FACTOR
                )
            if (
                growth_floor is None
                or not isinstance(prompt_tokens, int)
                or prompt_tokens < growth_floor
            ):
                return CompactionDecision(
                    trigger=None,
                    reason="soft_compaction_cooldown",
                    pressure_ratio=current_pressure,
                    soft_failure_threshold=required_failure_streak,
                    min_compactable_items=band.min_compactable_items,
                    hard_trigger_tokens=hard_trigger,
                )

    return CompactionDecision(
        trigger="compile_plateau",
        reason=band.name,
        pressure_ratio=current_pressure,
        soft_failure_threshold=required_failure_streak,
        min_compactable_items=band.min_compactable_items,
        hard_trigger_tokens=hard_trigger,
    )


# ---------------------------------------------------------------------------
# New: token estimation + message compression
# ---------------------------------------------------------------------------

def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Conservative token estimate: ~3.2 chars/token.

    Intentionally over-estimates to avoid underflow into truncation.
    To be calibrated post-launch against real eval_rollouts.jsonl tokenizer counts;
    if deviation > 20% from actual, switch to tokenizer-based estimation.
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total_chars += len(part.get("text", ""))
                elif isinstance(part, str):
                    total_chars += len(part)
        role = msg.get("role", "")
        total_chars += len(role) + 10  # role overhead + separators
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                total_chars += len(str(fn.get("name", "")))
                total_chars += len(str(fn.get("arguments", "")))
    return max(1, int(total_chars / CHARS_PER_TOKEN))


_COMPRESSED_HISTORY_RE = re.compile(
    r"<compressed_history>\s*(.*?)\s*</compressed_history>",
    re.DOTALL,
)


def _extract_summary_lines(content: str) -> list[str]:
    """Extract existing summary lines from a <compressed_history> block."""
    m = _COMPRESSED_HISTORY_RE.search(content)
    if not m:
        return []
    return [line for line in m.group(1).splitlines() if line.strip()]


def summarize_turn_group(
    turn_group: list[dict[str, Any]],
    record: TurnRecord | None,
    turn_index: int,
) -> str:
    """Compress one turn (assistant + tool messages) into a single summary line.

    Differentiates by tool type per plan注意事项 #6:
    - compile_model: keep severity + first 2 signal names
    - read_file: omit entirely (model can re-read)
    - write_file/replace: OK or FAIL only

    Multi tool_call turns merge into one line per plan注意事项 #10:
    e.g. ``T3: replace(model.py) OK → compile_model: 2 failures [isolated_part, real_overlap]``
    """
    tool_summaries: list[str] = []

    for msg in turn_group:
        role = msg.get("role", "")
        if role != "tool":
            continue
        content = msg.get("content", "")

        # Try to detect tool name from content or from the corresponding record
        if "<compile_signals>" in content:
            # compile_model result — extract severity and first signals
            if "failure" in content.lower():
                signals = re.findall(r'kind="([^"]+)"', content)
                sig_text = ", ".join(signals[:2]) if signals else "unknown"
                tool_summaries.append(f"compile_model: FAIL [{sig_text}]")
            elif "warning" in content.lower():
                tool_summaries.append("compile_model: OK+warnings")
            else:
                tool_summaries.append("compile_model: OK")
        elif '"output"' in content or '"error"' in content:
            # Generic tool result
            if '"error"' in content and "null" not in content.split('"error"')[1][:10]:
                tool_summaries.append("tool: FAIL")
            else:
                tool_summaries.append("tool: OK")

    if not tool_summaries and record:
        tc_names = [tc.get("name", "?") for tc in record.tool_calls]
        tool_summaries = [f"{n}()" for n in tc_names]

    summary = " → ".join(tool_summaries) if tool_summaries else "no tools"
    return f"T{turn_index}: {summary}"


def compact_messages(
    messages: list[dict[str, Any]],
    turn_records: list[TurnRecord],
    full_turns_to_keep: int,
) -> list[dict[str, Any]]:
    """Compress old turns into a <compressed_history> summary block.

    Turn boundary: each ``role="assistant"`` message starts a new turn.

    Multi-compaction support: prior ``<compressed_history>`` blocks are detected
    and *extended* (not duplicated). The block is always a single user-role message
    placed right before the first retained turn.

    Uses ``user`` role for the compressed block for TITO compatibility
    (completion_mask=False, trainer won't compute gradients on it).
    """
    # 1. Find the first assistant message (conversation start)
    first_assistant = next(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        len(messages),
    )

    # 2. Separate prefix from any existing compressed_history block
    existing_summary_lines: list[str] = []
    fixed_prefix: list[dict[str, Any]] = []
    for msg in messages[:first_assistant]:
        if (
            msg.get("role") == "user"
            and "<compressed_history>" in (msg.get("content") or "")
        ):
            existing_summary_lines = _extract_summary_lines(msg["content"])
        else:
            fixed_prefix.append(msg)

    conversation = messages[first_assistant:]

    # 3. Group conversation into turns (each starts with an assistant message)
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for msg in conversation:
        if msg.get("role") == "assistant" and current:
            turns.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append(current)

    # 4. Decide how many turns to compress
    compress_count = max(0, len(turns) - full_turns_to_keep)
    if compress_count == 0:
        return messages

    # Safety: turn count should roughly align with turn_records
    if len(turns) > len(turn_records) + 2:
        return messages  # mismatch fallback — don't corrupt

    # 5. Summarize old turns
    for i in range(compress_count):
        turn_group = turns[i]
        record_idx = len(existing_summary_lines)
        record = turn_records[record_idx] if record_idx < len(turn_records) else None
        line = summarize_turn_group(turn_group, record, turn_index=record_idx)
        existing_summary_lines.append(line)

    # 6. Rebuild: prefix + merged compressed block + retained turns
    summary_content = (
        "<compressed_history>\n"
        + "\n".join(existing_summary_lines)
        + "\n</compressed_history>"
    )
    result = list(fixed_prefix)
    result.append({"role": "user", "content": summary_content})
    for turn_group in turns[compress_count:]:
        result.extend(turn_group)

    return result
