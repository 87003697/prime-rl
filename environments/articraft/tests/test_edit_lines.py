"""Tests for EditLinesTool (v2 tool set)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent.tools.edit_lines import EditLinesTool


def _write_scaffold(script_path: Path, *, editable_code: str) -> None:
    script_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                'DEFAULT_NAME = "draft_model"',
                "",
                "# >>> USER_CODE_START",
                editable_code.rstrip(),
                "# >>> USER_CODE_END",
                "",
                "UNCHANGED_SENTINEL = True",
                "",
            ]
        ),
        encoding="utf-8",
    )


async def _run_edit(script_path: Path, params: dict) -> object:
    tool = EditLinesTool()
    invocation = await tool.build(params)
    invocation.bind_file_path(str(script_path))
    return await invocation.execute()


SAMPLE_CODE = """\
def build_object_model():
    x = 1
    delete_me = True
    return x

def run_tests():
    return None"""


def test_edit_single_line(tmp_path: Path) -> None:
    script_path = tmp_path / "model.py"
    _write_scaffold(script_path, editable_code=SAMPLE_CODE)

    result = asyncio.run(
        _run_edit(script_path, {"start_line": 2, "end_line": 2, "new_content": "    x = 42"})
    )
    assert result.error is None
    assert "Edited lines 2-2" in result.output

    updated = script_path.read_text(encoding="utf-8")
    assert "x = 42" in updated
    assert "x = 1" not in updated


def test_edit_multi_line(tmp_path: Path) -> None:
    script_path = tmp_path / "model.py"
    _write_scaffold(script_path, editable_code=SAMPLE_CODE)

    result = asyncio.run(
        _run_edit(
            script_path,
            {
                "start_line": 2,
                "end_line": 3,
                "new_content": "    x = 42\n    y = 99",
            },
        )
    )
    assert result.error is None
    updated = script_path.read_text(encoding="utf-8")
    assert "x = 42" in updated
    assert "y = 99" in updated
    assert "delete_me" not in updated


def test_invalid_range(tmp_path: Path) -> None:
    script_path = tmp_path / "model.py"
    _write_scaffold(script_path, editable_code=SAMPLE_CODE)

    result = asyncio.run(
        _run_edit(script_path, {"start_line": 50, "end_line": 60, "new_content": "x"})
    )
    assert result.error is not None
    assert "Invalid range" in result.error


def test_delete_lines(tmp_path: Path) -> None:
    script_path = tmp_path / "model.py"
    _write_scaffold(script_path, editable_code=SAMPLE_CODE)

    result = asyncio.run(
        _run_edit(script_path, {"start_line": 3, "end_line": 3, "new_content": ""})
    )
    assert result.error is None
    updated = script_path.read_text(encoding="utf-8")
    assert "delete_me" not in updated


def test_syntax_error_reported(tmp_path: Path) -> None:
    script_path = tmp_path / "model.py"
    _write_scaffold(script_path, editable_code=SAMPLE_CODE)

    result = asyncio.run(
        _run_edit(
            script_path,
            {"start_line": 1, "end_line": 1, "new_content": "def foo("},
        )
    )
    assert result.compilation["status"] == "error"


def test_scaffold_preserved(tmp_path: Path) -> None:
    """Edits must not corrupt scaffold markers."""
    script_path = tmp_path / "model.py"
    _write_scaffold(script_path, editable_code=SAMPLE_CODE)

    asyncio.run(
        _run_edit(script_path, {"start_line": 2, "end_line": 2, "new_content": "    x = 42"})
    )
    updated = script_path.read_text(encoding="utf-8")
    assert "# >>> USER_CODE_START" in updated
    assert "# >>> USER_CODE_END" in updated
    assert "UNCHANGED_SENTINEL = True" in updated
