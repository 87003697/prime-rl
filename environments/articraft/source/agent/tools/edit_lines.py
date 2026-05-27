"""EditLines tool — 按行号编辑代码（v2 工具集新增）。

背景：
  RL rollout 中模型频繁因 replace 工具的精确文本匹配失败而浪费步数。
  edit_lines 提供一条基于行号的替代路径：模型先通过 read_file 看到带 L{n}: 前缀
  的行号，再用 edit_lines(start_line, end_line, new_content) 直接替换行范围，
  完全绕过精确文本匹配。

  该工具仅在 tools_version="v2" 时注册到 ToolRegistry，v1 不受影响。

工作流：
  1. read_file(path="model.py") → 输出带行号的可编辑区域代码
  2. edit_lines(start_line=N, end_line=M, new_content=...) → 替换 [N, M] 行
  3. 写入后自动执行 Python 语法检查（compile()），结果附在 ToolResult.compilation 中

行号约定：
  - 1-indexed，start_line 和 end_line 均为闭区间
  - 行号对应可编辑区域（USER_CODE_START / USER_CODE_END 标记之间），不是全文件行号
  - new_content 可以比原始行范围多或少行（支持插入/删除）
  - new_content 为空字符串时删除指定行
"""

from __future__ import annotations

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


class EditLinesParams(ToolParamsModel):
    start_line: int
    end_line: int
    new_content: str


class EditLinesInvocation(BoundFileToolInvocation[EditLinesParams, str]):
    """按行号范围替换可编辑区域代码的 Invocation 实现。

    执行流程：
      读取全文件 → 提取可编辑区域 → 按行号切片替换 → 语法检查 → 写回文件。
    """

    def get_description(self) -> str:
        return f"Edit lines {self.params.start_line}-{self.params.end_line}"

    async def execute(self) -> ToolResult:
        if not self.file_path:
            return ToolResult(error="file_path is required")

        async with aiofiles.open(self.file_path, mode="r") as f:
            full_code = await f.read()

        # 提取 USER_CODE_START / USER_CODE_END 之间的可编辑区域
        editable = extract_editable_code(full_code)
        lines = editable.splitlines()
        total = len(lines)
        start, end = self.params.start_line, self.params.end_line

        # 边界校验：start_line ≥ 1，end_line ≥ start_line，start_line ≤ 总行数
        if start < 1 or end < start or start > total:
            return ToolResult(
                error=f"Invalid range {start}-{end} (file has {total} lines)"
            )
        # end_line 超出总行数时 clamp 到末尾（宽容处理，避免 off-by-one 导致错误）
        end = min(end, total)

        # 核心操作：用 new_content 的行替换 [start-1, end) 切片
        # 空 new_content → 删除行；行数不等 → 文件长度会变化
        new_lines = self.params.new_content.splitlines() if self.params.new_content else []
        lines[start - 1 : end] = new_lines

        # 将修改后的可编辑区域写回全文件（保留 scaffold 头尾不变）
        new_full = replace_editable_code(full_code, "\n".join(lines) + "\n")
        # 语法检查：compile() 执行，错误行号映射回可编辑区域行号
        validation = validate_python_syntax(new_full, self.file_path)

        async with aiofiles.open(self.file_path, mode="w") as f:
            await f.write(new_full)

        return ToolResult(output=f"Edited lines {start}-{end}", compilation=validation)


class EditLinesTool(BaseDeclarativeTool):
    """edit_lines 工具的声明式定义——生成 OpenAI-format schema 供 LLM 调用。"""

    def __init__(self) -> None:
        schema = make_tool_schema(
            name="edit_lines",
            description=(
                "Edit specific lines in the editable code section by line number.\n\n"
                "Line numbers correspond to the `L{n}:` prefix from "
                '`read_file(path="model.py")` output.\n\n'
                "Workflow:\n"
                '1. `read_file(path="model.py", offset=N, limit=M)` to see target lines\n'
                "2. `edit_lines(start_line=N, end_line=N+K, new_content=...)` to replace them\n\n"
                "Both start_line and end_line are inclusive (1-indexed). "
                "new_content replaces the specified line range entirely — "
                "it may have more or fewer lines than the original."
            ),
            parameters={
                "start_line": {
                    "type": "integer",
                    "description": "First line to replace (1-indexed, inclusive).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to replace (1-indexed, inclusive).",
                },
                "new_content": {
                    "type": "string",
                    "description": (
                        "Replacement content. May have more/fewer lines than the original. "
                        "Empty string deletes the lines."
                    ),
                },
            },
            required=["start_line", "end_line", "new_content"],
        )
        super().__init__("edit_lines", schema)

    async def build(self, params: dict) -> EditLinesInvocation:
        validated = validate_tool_params(EditLinesParams, params)
        return EditLinesInvocation(validated)
