"""Tests for ReplaceV2Tool (v2 tool set — closest-match error hints)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent.tools.replace_v2 import ReplaceV2Tool


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


async def _run_replace(script_path: Path, params: dict) -> object:
    tool = ReplaceV2Tool()
    invocation = await tool.build(params)
    invocation.bind_file_path(str(script_path))
    return await invocation.execute()


SAMPLE_CODE = """\
def build_object_model():
    x = 1
    y = 2
    return x + y

def run_tests():
    return None"""


def test_exact_match_still_works(tmp_path: Path) -> None:
    script_path = tmp_path / "model.py"
    _write_scaffold(script_path, editable_code=SAMPLE_CODE)

    result = asyncio.run(
        _run_replace(
            script_path,
            {"old_string": "    x = 1", "new_string": "    x = 42"},
        )
    )
    assert result.error is None
    assert "Code edited successfully" in result.output
    updated = script_path.read_text(encoding="utf-8")
    assert "x = 42" in updated


def test_closest_match_on_failure(tmp_path: Path) -> None:
    script_path = tmp_path / "model.py"
    _write_scaffold(script_path, editable_code=SAMPLE_CODE)

    result = asyncio.run(
        _run_replace(
            script_path,
            {"old_string": "    x = 1\n    y = 3", "new_string": "    x = 99"},
        )
    )
    assert result.error is not None
    assert "Closest match" in result.error
    assert "L" in result.error
    assert "edit_lines" in result.error


def test_no_match_at_all(tmp_path: Path) -> None:
    script_path = tmp_path / "model.py"
    _write_scaffold(script_path, editable_code=SAMPLE_CODE)

    result = asyncio.run(
        _run_replace(
            script_path,
            {"old_string": "completely_unrelated_string_xyz_123", "new_string": "x"},
        )
    )
    assert result.error is not None
    assert "Could not find" in result.error


def test_empty_old_string_on_empty_section(tmp_path: Path) -> None:
    script_path = tmp_path / "model.py"
    _write_scaffold(script_path, editable_code="")

    result = asyncio.run(
        _run_replace(
            script_path,
            {"old_string": "", "new_string": "x = 1\n"},
        )
    )
    assert result.error is None
    assert "Code edited successfully" in result.output


def test_empty_old_string_on_non_empty_section(tmp_path: Path) -> None:
    script_path = tmp_path / "model.py"
    _write_scaffold(script_path, editable_code=SAMPLE_CODE)

    result = asyncio.run(
        _run_replace(
            script_path,
            {"old_string": "", "new_string": "x = 1\n"},
        )
    )
    assert result.error is not None
    assert "old_string cannot be empty" in result.error


def test_allow_multiple(tmp_path: Path) -> None:
    script_path = tmp_path / "model.py"
    code = "a = 1\na = 1\na = 1\n"
    _write_scaffold(script_path, editable_code=code)

    result = asyncio.run(
        _run_replace(
            script_path,
            {"old_string": "a = 1", "new_string": "a = 2", "allow_multiple": True},
        )
    )
    assert result.error is None
    assert "3 occurrences" in result.output
    updated = script_path.read_text(encoding="utf-8")
    assert updated.count("a = 2") == 3
    assert "a = 1" not in updated


def test_multiple_occurrences_rejected_by_default(tmp_path: Path) -> None:
    script_path = tmp_path / "model.py"
    code = "a = 1\na = 1\n"
    _write_scaffold(script_path, editable_code=code)

    result = asyncio.run(
        _run_replace(
            script_path,
            {"old_string": "a = 1", "new_string": "a = 2"},
        )
    )
    assert result.error is not None
    assert "appears 2 times" in result.error


def test_scaffold_preserved(tmp_path: Path) -> None:
    script_path = tmp_path / "model.py"
    _write_scaffold(script_path, editable_code=SAMPLE_CODE)

    asyncio.run(
        _run_replace(
            script_path,
            {"old_string": "    x = 1", "new_string": "    x = 42"},
        )
    )
    updated = script_path.read_text(encoding="utf-8")
    assert "# >>> USER_CODE_START" in updated
    assert "# >>> USER_CODE_END" in updated
    assert "UNCHANGED_SENTINEL = True" in updated
    assert 'DEFAULT_NAME = "draft_model"' in updated
