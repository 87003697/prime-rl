"""ReplaceV2 — replace 工具的增强版，匹配失败时提供 closest-match 错误提示（v2 工具集）。

设计要点：
  - 工具名仍为 "replace"，LLM 侧 schema 与 v1 完全一致，零迁移成本
  - 独立新文件，不修改原 edit_code.py（v1 代码路径零改动，回滚只需改 TOML 一行）
  - 匹配成功时行为与 v1 ReplaceTool 完全一致
  - 匹配失败时（v2 增强）：
    1. 用滑动窗口 + difflib.SequenceMatcher 找到最相似的代码区域
    2. 在错误信息中展示实际代码（带行号）+ 相似度百分比
    3. 同时建议使用 edit_lines 按行号编辑，为模型提供替代修复路径

与 env.py _dispatch_tool 的交互：
  env.py 第 452-484 行有 replace 的 empty old_string 拦截，在 v2 模式下仍会先于
  ReplaceV2Invocation.execute() 执行。ReplaceV2Invocation 内部也有一份相同逻辑。
  两处返回相似错误，不是 bug，但 v2 的内部分支永远不会被触发——这是为了保持 v1 零修改
  原则的有意设计取舍。
"""

from __future__ import annotations

import difflib

import aiofiles

from agent.tools.base import (
    BaseDeclarativeTool,
    BoundFileToolInvocation,
    ToolParamsModel,
    ToolResult,
    make_tool_schema,
    validate_tool_params,
)
from agent.tools.code_region import (
    extract_editable_code,
    replace_editable_code,
    validate_python_syntax,
)


def _find_closest_match(
    old_string: str, editable_code: str, *, context_lines: int = 1,
) -> dict | None:
    """在可编辑区域中找到与 old_string 最相似的代码片段。

    算法：
      将 old_string 和 editable_code 都按行拆分，用等长滑动窗口遍历代码，
      对每个窗口位置计算 SequenceMatcher.ratio()（0~1 的相似度）。
      找到相似度最高的位置后，附加 context_lines 行上下文，返回带行号的文本。

    性能保护：
      - 代码超过 500 行或 old_string 超过 50 行时跳过（避免 O(n*m) 开销过大）
      - 相似度超过 0.85 时提前退出（"够好了"策略）
      - 最低阈值 0.3——低于此值说明完全不相关，返回 None

    返回值：
      匹配成功: {"start_line": int, "end_line": int, "similarity": float, "actual_text": str}
      匹配失败: None
    """
    old_lines = old_string.splitlines()
    code_lines = editable_code.splitlines()
    if not old_lines or not code_lines:
        return None
    # 大文件或大片段时放弃搜索，避免影响 rollout 延迟
    if len(code_lines) > 500 or len(old_lines) > 50:
        return None

    window_size = max(1, len(old_lines))
    best_ratio = 0.0
    best_start = 0

    # 滑动窗口：窗口大小 = old_string 行数，步长 = 1
    for i in range(max(1, len(code_lines) - window_size + 1)):
        window = code_lines[i : i + window_size]
        ratio = difflib.SequenceMatcher(None, old_lines, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
            # 相似度已经很高，提前退出节省计算
            if ratio > 0.85:
                break

    # 相似度太低说明没有有意义的匹配
    if best_ratio < 0.3:
        return None

    # 附加上下文行，让模型看到匹配区域的周边代码
    ctx_start = max(0, best_start - context_lines)
    ctx_end = min(len(code_lines), best_start + window_size + context_lines)
    actual_lines = code_lines[ctx_start:ctx_end]

    # 返回的 actual_text 带 L{n}: 前缀，与 read_file 输出格式一致，
    # 方便模型直接复制或用 edit_lines 的行号
    return {
        "start_line": ctx_start + 1,
        "end_line": ctx_end,
        "similarity": best_ratio,
        "actual_text": "\n".join(
            f"L{ctx_start + 1 + j}: {line}" for j, line in enumerate(actual_lines)
        ),
    }


class ReplaceV2Params(ToolParamsModel):
    """与 v1 ReplaceTool 参数 schema 完全一致，确保 LLM 侧无需任何调整。"""

    old_string: str
    new_string: str
    instruction: str | None = None
    allow_multiple: bool = False


class ReplaceV2Invocation(BoundFileToolInvocation[ReplaceV2Params, str]):
    """Replace 的 v2 实现——匹配失败时附带 closest-match 提示。

    执行流程：
      1. 读取全文件 → 提取可编辑区域
      2. 空 old_string 特殊处理：仅在可编辑区域为空时允许（用于初始化代码）
      3. 精确匹配检查：
         - 匹配成功 → 执行替换（与 v1 行为一致）
         - 匹配失败（v2 增强）→ 调用 _find_closest_match 找最相似片段，
           在错误信息中展示实际代码 + 行号 + edit_lines 用法建议
      4. 多次出现保护：默认要求 old_string 唯一，allow_multiple=True 时替换全部
      5. 写回文件前执行 Python 语法检查
    """

    def get_description(self) -> str:
        preview = self.params.old_string[:50]
        if len(self.params.old_string) > 50:
            preview += "..."
        return f"Replace text: '{preview}'"

    async def execute(self) -> ToolResult:
        if not self.file_path:
            return ToolResult(error="file_path is required")

        async with aiofiles.open(self.file_path, mode="r") as f:
            full_code = await f.read()

        editable_code = extract_editable_code(full_code)

        # --- 空 old_string：初始化可编辑区域 ---
        # 仅在可编辑区域完全为空时允许（scaffold 刚创建，尚无用户代码）
        if not self.params.old_string:
            if editable_code.strip():
                return ToolResult(
                    error=(
                        "old_string cannot be empty unless the editable section is empty. "
                        "Please provide the exact string to replace."
                    )
                )
            new_code = replace_editable_code(full_code, self.params.new_string)
            validation = validate_python_syntax(new_code, self.file_path)
            async with aiofiles.open(self.file_path, mode="w") as f:
                await f.write(new_code)
            return ToolResult(output="Code edited successfully", compilation=validation)

        # --- v2 核心增强：匹配失败时提供 closest-match ---
        # v1 只返回 "Could not find..."，模型经常陷入盲目重试循环。
        # v2 展示最相似片段的实际代码和行号，给模型两条修复路径：
        #   路径 A: 复制展示的精确文本作为 old_string 重试 replace
        #   路径 B: 用 edit_lines(start_line=N, end_line=M, ...) 按行号编辑
        if self.params.old_string not in editable_code:
            match = _find_closest_match(self.params.old_string, editable_code)
            if match:
                return ToolResult(
                    error=(
                        f"Could not find old_string in the code. "
                        f"Closest match ({match['similarity']:.0%} similar) "
                        f"at lines {match['start_line']}-{match['end_line']}:\n"
                        f"{match['actual_text']}\n\n"
                        f"Copy exact text from above as old_string, or use "
                        f"edit_lines(start_line={match['start_line']}, "
                        f"end_line={match['end_line']}, new_content=...)."
                    )
                )
            # 相似度太低（<0.3）或文件太大/old_string 太长，退回到通用错误信息
            return ToolResult(
                error="Could not find the old_string in the code. "
                "Make sure the string matches exactly, including whitespace and indentation. "
                'Call read_file(path="model.py") to check current code.'
            )

        # --- 多次出现保护 ---
        # 默认要求 old_string 在可编辑区域中唯一出现，防止意外批量替换
        occurrences = editable_code.count(self.params.old_string)
        if occurrences > 1 and not self.params.allow_multiple:
            return ToolResult(
                error=f"The old_string appears {occurrences} times in the code. "
                "Please provide a longer, unique string, "
                "or use allow_multiple=true to replace all occurrences."
            )

        # --- 执行替换 ---
        if self.params.allow_multiple:
            new_editable = editable_code.replace(
                self.params.old_string, self.params.new_string
            )
        else:
            new_editable = editable_code.replace(
                self.params.old_string, self.params.new_string, 1
            )

        new_code = replace_editable_code(full_code, new_editable)
        validation = validate_python_syntax(new_code, self.file_path)

        async with aiofiles.open(self.file_path, mode="w") as f:
            await f.write(new_code)

        if self.params.allow_multiple and occurrences > 1:
            msg = f"Code edited successfully (replaced {occurrences} occurrences)"
        else:
            msg = "Code edited successfully"

        return ToolResult(output=msg, compilation=validation)


class ReplaceV2Tool(BaseDeclarativeTool):
    """replace 工具的 v2 声明式定义——带 closest-match 错误提示。

    工具名仍为 "replace"（不是 "replace_v2"），LLM 侧 schema 与 v1 完全一致。
    在 tools_version="v2" 时替代原 ReplaceTool 注册到 ToolRegistry。
    """

    def __init__(self) -> None:
        schema = make_tool_schema(
            name="replace",
            description=(
                "Replace text within the current editable code section.\n\n"
                "Matching behavior:\n"
                "- `old_string` must match EXACTLY, including whitespace, indentation, and newlines\n"
                "- If the editable section is empty, `old_string` may be empty to insert initial code\n"
                "- By default, `old_string` must appear exactly once\n"
                "- Set `allow_multiple=true` to replace all exact matches\n"
                "- If exact matching fails, the error will show the closest matching region "
                "with line numbers for easy correction"
            ),
            parameters={
                "old_string": {
                    "type": "string",
                    "description": (
                        "Exact literal text to find inside the editable code section. "
                        "Must match including whitespace and indentation."
                    ),
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text. May be empty to delete the matched text.",
                },
                "instruction": {
                    "type": "string",
                    "description": "Optional short description of the intended change.",
                },
                "allow_multiple": {
                    "type": "boolean",
                    "description": (
                        "If false (default), `old_string` must be unique. "
                        "If true, replace all exact matches."
                    ),
                },
            },
            required=["old_string", "new_string"],
        )
        super().__init__("replace", schema)

    async def build(self, params: dict) -> ReplaceV2Invocation:
        validated = validate_tool_params(ReplaceV2Params, params)
        return ReplaceV2Invocation(validated)
