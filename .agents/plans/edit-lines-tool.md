# Plan: edit_lines 工具 + replace 增强错误信息（最终版）

## 目标

1. **前置**：将 articraft 源码子集迁入 prime-rl，简化部署流程
2. **主体**：缓解 RL rollout 中工具调用失败率（replace 精确匹配是主要失败模式），通过 edit_lines + closest match 提供替代路径

---

## 设计核心：`tools_version` 版本化

```toml
[orchestrator.train.env.args]
tools_version = "v2"   # 不写或 "v1" → 原始行为完全不变
```

| | v1（默认，零修改） | v2（新特性） |
|---|---|---|
| 工具集 | read_file, replace, write_file, compile_model | read_file, **replace**(v2), write_file, compile_model, **edit_lines** |
| replace 错误 | 原始 "Could not find..." | closest match + 行号 + edit_lines 建议 |
| guidance | 原始 EDIT_RETRY_CONTENT | + edit_lines 提示 |
| 源码影响 | `edit_code.py` 完全不动 | 新文件 `replace_v2.py` + `edit_lines.py` |

**关键原则**：v1 的代码路径零修改。v2 全部是**新增文件**，不改已有文件逻辑。

---

## Part A: Articraft 源码迁入 prime-rl

### 当前 → 目标

```
当前: articraft-code.tar (S3) → cat|tar xf → /data/work/articraft/
目标: prime-rl 自带 source/ → s5cmd sync 已覆盖 → 零额外步骤
```

### 目标目录结构

```
prime-rl/environments/articraft/
├── source/                      ← 新增：articraft 代码子集（~10MB, ~860 files）
│   ├── agent/                   ← 工具、编译器、feedback
│   ├── sdk/                     ← SDK 文档 + profiles
│   ├── articraft/               ← core values/types
│   ├── data/
│   │   ├── categories/          ← 类别元数据 (449 files, 110KB)
│   │   └── system_prompts/      ← provider system prompts (93 files, 984KB)
│   ├── scaffold.py              ← 代码模板
│   └── pyproject.toml           ← 包定义 (editable install)
├── articraft_env/               ← 现有 RL 环境代码（不动）
├── tests/
└── pyproject.toml
```

### 实现步骤

- [ ] A1: **原样复制**代码子集
- [ ] A2: 修改 `scripts/envs/articraft.sh`
- [ ] A3: 验证 import 正常
- [ ] A4: 迁入 test_edit_code.py

### A1 执行方式

> **⚠️ A1 是纯 `cp -r` 复制，不修改任何源文件内容。所有代码改动都在 Part B 中进行。**

```bash
SRC=/Users/zhiyuanma/Desktop/codes/articraft
DST=/Users/zhiyuanma/Desktop/codes/prime-rl/environments/articraft/source

mkdir -p "$DST/data"
cp -r "$SRC/agent"      "$DST/agent"
cp -r "$SRC/sdk"        "$DST/sdk"
cp -r "$SRC/articraft"  "$DST/articraft"
cp -r "$SRC/data/categories"     "$DST/data/categories"
cp -r "$SRC/data/system_prompts" "$DST/data/system_prompts"
cp "$SRC/scaffold.py"    "$DST/scaffold.py"
cp "$SRC/pyproject.toml" "$DST/pyproject.toml"
find "$DST" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null
```

复制后的文件与 articraft 仓库源码**逐字节一致**（除 `__pycache__` 外）。

### A2: `scripts/envs/articraft.sh` 改动

```bash
# 改为：
ARTICRAFT_DIR="${PROJECT_DIR}/environments/articraft/source"

setup_ac_sync_code() {
    echo "  [env] Articraft source is at ${ARTICRAFT_DIR} (shipped with prime-rl)"
    if [ ! -d "${ARTICRAFT_DIR}/agent" ]; then
        echo "    ERROR: ${ARTICRAFT_DIR}/agent not found."
        exit 1
    fi
    echo "    OK ($(find ${ARTICRAFT_DIR} -type f | wc -l) files)"
}
```

---

## Part B: edit_lines + replace v2

### 文件变动总览

| 文件 | 操作 | 说明 |
|------|------|------|
| `source/agent/tools/edit_lines.py` | **新建** | EditLinesTool |
| `source/agent/tools/replace_v2.py` | **新建** | ReplaceV2Tool（集成原功能 + closest match） |
| `source/agent/tools/code_region.py` | 新增函数 | `validate_python_syntax()` 公共函数 |
| `source/agent/tools/__init__.py` | 修改 | 导出新工具 |
| `source/agent/tools/edit_code.py` | **不动** | v1 行为保持不变 |
| `articraft_env/schema.py` | 修改 | MUTATING_TOOL_NAMES 加 `"edit_lines"` |
| `articraft_env/env.py` | 修改 | `tools_version` 分支逻辑 |
| `articraft_env/guidance.py` | 修改 | 条件化 edit_lines 提示 |
| `configs/articraft/rl_articraft_kaola.toml` | 修改 | `tools_version = "v2"` |
| `tests/test_edit_lines.py` | **新建** | 测试 |

### 实现步骤

- [ ] B1: 修改 `source/agent/tools/code_region.py` — 新增 `validate_python_syntax()`
- [ ] B2: 新建 `source/agent/tools/edit_lines.py` — EditLinesTool
- [ ] B3: 新建 `source/agent/tools/replace_v2.py` — ReplaceV2Tool
- [ ] B4: 修改 `source/agent/tools/__init__.py` — 导出
- [ ] B5: 修改 `articraft_env/schema.py` — MUTATING_TOOL_NAMES
- [ ] B6: 修改 `articraft_env/env.py` — `tools_version` 分支
- [ ] B7: 修改 `articraft_env/guidance.py` — 条件化
- [ ] B8: 修改 TOML 配置
- [ ] B9: 新建测试
- [ ] B10: 运行测试

---

### B1: `source/agent/tools/code_region.py` — 新增公共函数

在文件末尾追加（不改已有代码）：

```python
def validate_python_syntax(full_code: str, filename: str) -> dict:
    """Validate syntax, mapping errors to editable-region line numbers."""
    try:
        compile(full_code, filename, "exec")
        return {"status": "success", "error": None}
    except SyntaxError as exc:
        editable_line = map_syntax_error_line_to_editable(full_code, exc.lineno)
        if editable_line is not None and editable_line != exc.lineno:
            error_msg = f"Syntax error: {exc.msg} (editable line {editable_line})"
        else:
            error_msg = f"Syntax error: {exc.msg} (line {exc.lineno})"
        return {"status": "error", "error": error_msg}
    except Exception as exc:
        return {"status": "error", "error": f"Validation error: {exc!s}"}
```

---

### B2: `source/agent/tools/edit_lines.py` — 新建

```python
"""EditLines tool - Edit specific lines by line number (v2 tool set)."""

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

    def get_description(self) -> str:
        return f"Edit lines {self.params.start_line}-{self.params.end_line}"

    async def execute(self) -> ToolResult:
        if not self.file_path:
            return ToolResult(error="file_path is required")

        async with aiofiles.open(self.file_path, mode="r") as f:
            full_code = await f.read()

        editable = extract_editable_code(full_code)
        lines = editable.splitlines()
        total = len(lines)
        start, end = self.params.start_line, self.params.end_line

        if start < 1 or end < start or start > total:
            return ToolResult(
                error=f"Invalid range {start}-{end} (file has {total} lines)"
            )
        end = min(end, total)

        new_lines = self.params.new_content.splitlines() if self.params.new_content else []
        lines[start - 1 : end] = new_lines

        new_full = replace_editable_code(full_code, "\n".join(lines) + "\n")
        validation = validate_python_syntax(new_full, self.file_path)

        async with aiofiles.open(self.file_path, mode="w") as f:
            await f.write(new_full)

        return ToolResult(output=f"Edited lines {start}-{end}", compilation=validation)


class EditLinesTool(BaseDeclarativeTool):

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
```

---

### B3: `source/agent/tools/replace_v2.py` — 新建

```python
"""ReplaceV2 — drop-in replacement for ReplaceTool with closest-match error hints (v2 tool set).

Tool name remains "replace" so the model schema is identical.
Integrates all original ReplaceTool/EditCodeInvocation functionality,
plus _find_closest_match on failure.
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
    """Find the code region most similar to old_string."""
    old_lines = old_string.splitlines()
    code_lines = editable_code.splitlines()
    if not old_lines or not code_lines:
        return None
    if len(code_lines) > 500 or len(old_lines) > 50:
        return None

    window_size = max(1, len(old_lines))
    best_ratio = 0.0
    best_start = 0

    for i in range(max(1, len(code_lines) - window_size + 1)):
        window = code_lines[i : i + window_size]
        ratio = difflib.SequenceMatcher(None, old_lines, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
            if ratio > 0.85:
                break

    if best_ratio < 0.3:
        return None

    ctx_start = max(0, best_start - context_lines)
    ctx_end = min(len(code_lines), best_start + window_size + context_lines)
    actual_lines = code_lines[ctx_start:ctx_end]

    return {
        "start_line": ctx_start + 1,
        "end_line": ctx_end,
        "similarity": best_ratio,
        "actual_text": "\n".join(
            f"L{ctx_start + 1 + j}: {line}" for j, line in enumerate(actual_lines)
        ),
    }


class ReplaceV2Params(ToolParamsModel):
    """Same schema as original ReplaceTool."""

    old_string: str
    new_string: str
    instruction: str | None = None
    allow_multiple: bool = False


class ReplaceV2Invocation(BoundFileToolInvocation[ReplaceV2Params, str]):
    """Replace with closest-match error hints."""

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

        # Empty old_string: initialize editable section
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

        # Check match — v2 enhancement: closest match on failure
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
            return ToolResult(
                error="Could not find the old_string in the code. "
                "Make sure the string matches exactly, including whitespace and indentation. "
                'Call read_file(path="model.py") to check current code.'
            )

        # Count occurrences
        occurrences = editable_code.count(self.params.old_string)
        if occurrences > 1 and not self.params.allow_multiple:
            return ToolResult(
                error=f"The old_string appears {occurrences} times in the code. "
                "Please provide a longer, unique string, "
                "or use allow_multiple=true to replace all occurrences."
            )

        # Perform replacement
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
    """Drop-in replacement for ReplaceTool with closest-match hints.

    Tool name is still "replace" — schema identical to v1.
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
```

---

### B4: `source/agent/tools/__init__.py`

```diff
 from agent.tools.edit_code import ReplaceTool
+from agent.tools.edit_lines import EditLinesTool
+from agent.tools.replace_v2 import ReplaceV2Tool
 from agent.tools.find_examples import FindExamplesTool
```

```diff
 __all__ = [
     ...
     "CompileModelTool",
+    "EditLinesTool",
     "FindExamplesTool",
     ...
     "ReplaceTool",
+    "ReplaceV2Tool",
     "WriteFileTool",
```

注意：`build_tool_registry()` 不改（它是 articraft 推理用的，始终用 v1 工具集）。

---

### B5: `articraft_env/schema.py`

```diff
-MUTATING_TOOL_NAMES = frozenset({"apply_patch", "replace", "write_file"})
+MUTATING_TOOL_NAMES = frozenset({"apply_patch", "replace", "write_file", "edit_lines"})
```

（`"edit_lines"` 在 set 里但未注册时零副作用，与 `"apply_patch"` 同理。）

---

### B6: `articraft_env/env.py` — `tools_version` 分支

```diff
     def __init__(
         self,
         ...
+        tools_version: str = "v1",
         **kwargs: Any,
     ) -> None:
         ...
+        self._tools_version = tools_version
+
         # -- articraft tool registry --
-        tools = [
-            ReadFileTool(editable_model_only=True),
-            ReplaceTool(),
-            WriteFileTool(),
-            CompileModelTool(),
-        ]
+        if tools_version == "v2":
+            from agent.tools.edit_lines import EditLinesTool
+            from agent.tools.replace_v2 import ReplaceV2Tool
+            tools = [
+                ReadFileTool(editable_model_only=True),
+                ReplaceV2Tool(),
+                WriteFileTool(),
+                CompileModelTool(),
+                EditLinesTool(),
+            ]
+        else:
+            tools = [
+                ReadFileTool(editable_model_only=True),
+                ReplaceTool(),
+                WriteFileTool(),
+                CompileModelTool(),
+            ]
+
         self.tool_registry = ToolRegistry(tools)
```

---

### B7: `articraft_env/guidance.py` — 新增 v2 函数

原有 `maybe_inject_edit_code_guidance` 不动。新增 v2 版本：

```diff
+def maybe_inject_edit_code_guidance_v2(
+    tool_name: str,
+    tool_error: str | None,
+    *,
+    already_injected: bool,
+) -> dict[str, Any] | None:
+    """V2 guidance: same logic as v1 + edit_lines hint."""
+    if tool_name != "replace":
+        return None
+    if not tool_error:
+        return None
+    if "Could not find" not in tool_error and "old_string" not in tool_error.lower():
+        return None
+    if already_injected:
+        return None
+    return {"role": "user", "content": (
+        "<edit_retry_guidance>\n"
+        "- Your last replace failed because `old_string` did not match the file exactly.\n"
+        '- Do NOT guess. Call `read_file(path="model.py")` again, then pick a smaller exact '
+        'snippet from the current editable code as `old_string` and retry.\n'
+        "- Alternatively, use `edit_lines(start_line=N, end_line=M, new_content=...)` "
+        "to edit by line number without needing to match exact text.\n"
+        "- Keep edits surgical.\n"
+        "</edit_retry_guidance>"
+    )}
```

`env.py` 在 `__init__` 中建立映射，`env_response` 中直接调用：

```diff
+from .guidance import compile_signal_signature, maybe_inject_edit_code_guidance, maybe_inject_edit_code_guidance_v2

     def __init__(self, ...):
         ...
+        # guidance 函数映射
+        if self._tools_version == "v2":
+            self._inject_edit_guidance = maybe_inject_edit_code_guidance_v2
+        else:
+            self._inject_edit_guidance = maybe_inject_edit_code_guidance
```

```diff
     # env_response() 中调用处
-            guidance_msg = maybe_inject_edit_code_guidance(
+            guidance_msg = self._inject_edit_guidance(
                 tool_name=_tc_name(tc),
                 tool_error=result.error,
                 already_injected=rollout.edit_retry_injected,
             )
```

---

### B8: TOML 配置

```toml
# configs/articraft/rl_articraft_kaola.toml
[orchestrator.train.env.args]
tools_version = "v2"
```

---

### B9: 测试

```python
# tests/test_edit_lines.py

def test_edit_single_line(tmp_path):
    result = await _run_edit(path, {"start_line": 2, "end_line": 2, "new_content": "replaced"})
    assert result.error is None
    assert "Edited lines 2-2" in result.output

def test_invalid_range(tmp_path):
    result = await _run_edit(path, {"start_line": 5, "end_line": 6, "new_content": "x"})
    assert "Invalid range" in result.error

def test_delete_lines(tmp_path):
    result = await _run_edit(path, {"start_line": 2, "end_line": 2, "new_content": ""})
    assert "delete_me" not in path.read_text()

def test_syntax_error(tmp_path):
    result = await _run_edit(path, {"start_line": 1, "end_line": 2, "new_content": "def foo(\n"})
    assert result.compilation["status"] == "error"

# tests/test_replace_v2.py

def test_closest_match_on_failure(tmp_path):
    # old_string 稍有偏差
    result = await _run_replace_v2(path, {"old_string": "almost_right", "new_string": "x"})
    assert "Closest match" in result.error
    assert "L" in result.error  # 包含行号

def test_exact_match_still_works(tmp_path):
    result = await _run_replace_v2(path, {"old_string": "exact", "new_string": "new"})
    assert result.error is None
```

---

## 工具集

| `tools_version` | 工具 | 数量 | 行为 |
|-----------------|------|------|------|
| `"v1"` (默认) | read_file, replace, write_file, compile_model | 4 | 与当前代码逐字节一致 |
| `"v2"` | read_file, replace(v2), write_file, compile_model, edit_lines | 5 | closest match + edit_lines |

---

## 执行顺序

```
Part A (cp -r 复制 + setup 脚本)  →  Part B (新建 v2 文件 + env.py 分支)  →  测试  →  提交
```

## 状态

**当前阶段**: 最终版 — 等待确认执行

# 评审（终审）

**整体判断：可以执行。**

### 遗留小问题

#### 1. `_dispatch_tool` 的 replace 拦截与 ReplaceV2 重复 [P3]

`env.py` 第 452-484 行有 replace 的 empty `old_string` 拦截，v2 时仍会先于 `ReplaceV2Invocation.execute()` 执行。`ReplaceV2Invocation` 内部也有一份（第 365-377 行）。不是 bug（两处返回相似错误），但 v2 的内部逻辑永远不会被触发。放着不管即可——符合 v1 零修改原则。

#### 2. B9 测试是伪代码 [P3]

测试缺少 scaffold 辅助函数、import、async 处理等。执行时需补全（早期版本里有完整的 `_write_scaffold` 和 `_run_edit` 辅助函数可复用）。

### 设计亮点

- **`replace_v2.py` 独立新建**：v1 代码零修改，回滚只需改 TOML 一行
- **`tools_version` 版本化**：工具集 + guidance + 错误信息作为一个整体切换，比特性列表更清晰
- **`ReplaceV2Tool` 保持 `name="replace"`**：模型侧 schema 完全一致
- **`maybe_inject_edit_code_guidance_v2` 独立函数**：guidance 也遵循 v1 不动原则，通过 `self._inject_edit_guidance` 映射切换
- **`tools_version == "v2"` 精确匹配**：避免字符串比较陷阱

### 总结

| 项 | 状态 |
|---|---|
| 内部一致性 | ✅ 无矛盾 |
| v1 零修改保证 | ✅ 新文件 + env.py/guidance.py 分支 |
| 测试与代码对齐 | ✅ 断言匹配 |
| 可执行性 | ✅ B1-B10 可按序执行 |
| 遗留问题 | 2 个 P3，不阻塞 |

可以开始了。