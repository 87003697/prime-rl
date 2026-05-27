"""Guidance injection — mirrors harness_guidance.py GuidanceInjector.

Phase 2 implements:
  - compile_signal_signature() — for failure streak detection (Feature #1)
  - maybe_inject_edit_code_guidance() — edit_retry guidance (Feature #2)

Future: exact_geometry / baseline_qc guidance (requires scan_code_contracts).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def compile_signal_signature(bundle_dict: dict[str, Any]) -> str:
    """SHA-1 of bundle dict — mirrors CompileFeedbackLoop._compile_signal_signature."""
    raw = json.dumps(bundle_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


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
    """Return a user-role guidance message if a replace failed on old_string mismatch.

    Mirrors GuidanceInjector.maybe_inject_edit_code_guidance().
    One-shot per rollout: caller passes ``already_injected`` from Rollout state.
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


# V2 guidance 文本：与 v1 相同的 read_file → retry 建议，额外增加 edit_lines 提示。
# 这行额外提示是 v2 相对 v1 的唯一 guidance 差异——让模型知道可以用行号编辑
# 来绕过精确文本匹配，降低 replace 反复失败时的步数浪费。
EDIT_RETRY_CONTENT_V2 = (
    "<edit_retry_guidance>\n"
    "- Your last replace failed because `old_string` did not match the file exactly.\n"
    '- Do NOT guess. Call `read_file(path="model.py")` again, then pick a smaller exact '
    'snippet from the current editable code as `old_string` and retry.\n'
    "- Alternatively, use `edit_lines(start_line=N, end_line=M, new_content=...)` "
    "to edit by line number without needing to match exact text.\n"
    "- Keep edits surgical.\n"
    "</edit_retry_guidance>"
)


def maybe_inject_edit_code_guidance_v2(
    tool_name: str,
    tool_error: str | None,
    *,
    already_injected: bool,
) -> dict[str, Any] | None:
    """V2 edit_retry guidance：与 v1 相同的触发逻辑 + edit_lines 使用建议。

    触发条件（与 v1 一致）：
      - tool_name == "replace"
      - tool_error 非空且包含 "Could not find" 或 "old_string"
      - 本 rollout 尚未注入过（one-shot，避免重复注入浪费 context）

    V2 差异点：
      guidance 文本中多一行 edit_lines 的用法提示，告知模型可以按行号编辑。

    注入方式：user role message，用于 TITO 兼容（completion_mask=False，不产生梯度）。
    由 env.py 中 self._inject_edit_guidance 动态映射到此函数（v2 时）或 v1 版本。
    """
    if tool_name != "replace":
        return None
    if not tool_error:
        return None
    if "Could not find" not in tool_error and "old_string" not in tool_error.lower():
        return None
    if already_injected:
        return None
    return {"role": "user", "content": EDIT_RETRY_CONTENT_V2}
