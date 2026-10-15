from pathlib import Path
from textwrap import dedent

from meta.notebook_tools.cli import sync
from meta.notebook_tools.helpers import generate_exercise_notebooks


def test_sync(tmpdir):
    tmpdir = Path(tmpdir)
    base_notebook = Path(__file__).parent / "sample_notebook.ipynb"
    expected_normal = base_notebook.with_name("sample_notebook_normal.ipynb")
    expected_hard = base_notebook.with_name("sample_notebook_hard.ipynb")

    input_notebook = tmpdir / "sample_notebook.ipynb"
    input_notebook.write_text(base_notebook.read_text("utf-8"), encoding="utf-8")
    output_normal = tmpdir / "sample_notebook_normal.ipynb"
    output_hard = tmpdir / "sample_notebook_hard.ipynb"

    sync([input_notebook])

    for output, expected in [(output_normal, expected_normal), (output_hard, expected_hard)]:
        for line1, line2 in zip(
            output.read_text("utf-8").splitlines(), expected.read_text("utf-8").splitlines()
        ):
            assert line1 == line2
        assert len(output.read_text("utf-8")) == len(expected.read_text("utf-8"))


def make_notebook(source: str) -> dict:
    lines = dedent(source).lstrip("\n").splitlines(keepends=True)
    return {"cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "source": lines}]}


def exercise_code(notebook: dict, label: str) -> str:
    return "".join(generate_exercise_notebooks(notebook)[label]["cells"][0]["source"])


def test_one_hidden_word_is_singular():
    notebook = make_notebook(
        """
        # Hide: all
        x
        """
    )
    assert exercise_code(notebook, "normal") == "...  # TODO: ~1 word\n"
