from pathlib import Path
from textwrap import dedent

import pytest
import typer

from meta.notebook_tools.cli import sync
from meta.notebook_tools.helpers import generate_exercise_notebooks, notebook_to_str


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


def solution_code(notebook: dict, label: str) -> str:
    return "".join(generate_exercise_notebooks(notebook)[label]["cells"][1]["source"])


def test_blank_in_every_notebook():
    notebook = make_notebook(
        """
        # Hide: hard
        x = 1
        # Hide: none
        # Blank: f(x)
        y = f(x) + 1
        """
    )
    assert exercise_code(notebook, "normal") == "x = 1\ny = ________ + 1\n"
    assert exercise_code(notebook, "hard") == "...  # TODO: ~2 words\ny = ________ + 1\n"
    assert "y = f(x) + 1\n" in solution_code(notebook, "normal")
    assert "Blank" not in solution_code(notebook, "normal")


def test_blank_only_generates_normal_notebook():
    notebook = make_notebook(
        """
        # Blank: f(x)
        y = f(x)
        """
    )
    assert set(generate_exercise_notebooks(notebook)) == {"normal"}
    assert exercise_code(notebook, "normal") == "y = ________\n"
    assert "y = f(x)\n" in solution_code(notebook, "normal")


def test_blank_per_notebook():
    notebook = make_notebook(
        """
        # Blank[normal]: run_with_cache
        # Blank[hard]: model.run_with_cache(tokens)
        logits, cache = model.run_with_cache(tokens)
        # Hide: hard
        x = 1
        """
    )
    assert exercise_code(notebook, "normal") == "logits, cache = model.________(tokens)\nx = 1\n"
    assert exercise_code(notebook, "hard") == "logits, cache = ________\n...  # TODO: ~2 words\n"


def test_blank_in_hard_only_shows_line_in_normal():
    notebook = make_notebook(
        """
        # Blank[hard]: f(x)
        y = f(x)
        # Hide: hard
        x = 1
        """
    )
    assert exercise_code(notebook, "normal") == "y = f(x)\nx = 1\n"
    assert exercise_code(notebook, "hard") == "y = ________\n...  # TODO: ~2 words\n"


def test_blank_of_hidden_line_is_skipped():
    notebook = make_notebook(
        """
        # Hide: hard
        # Blank: f(x)
        y = f(x)
        # Hide: none
        """
    )
    assert exercise_code(notebook, "normal") == "y = ________\n"
    assert exercise_code(notebook, "hard") == "...  # TODO: ~3 words\n"


def test_blank_label_creates_notebook():
    notebook = make_notebook(
        """
        # Blank[hard]: f(x)
        y = f(x)
        """
    )
    assert set(generate_exercise_notebooks(notebook)) == {"normal", "hard"}
    assert exercise_code(notebook, "normal") == "y = f(x)\n"
    assert exercise_code(notebook, "hard") == "y = ________\n"


def test_one_hidden_word_is_singular():
    notebook = make_notebook(
        """
        # Hide: all
        x
        """
    )
    assert exercise_code(notebook, "normal") == "...  # TODO: ~1 word\n"


def test_blank_whole_line_before_hidden_lines():
    notebook = make_notebook(
        """
        # Blank: y = f(x)
        y = f(x)
        # Hide: all
        z = g(y)
        """
    )
    assert exercise_code(notebook, "normal") == "________\n...  # TODO: ~3 words\n"


@pytest.mark.parametrize(
    "source, error",
    [
        ("# Blank: g(x)\ny = f(x)\n", "found 0 times"),
        ("# Blank: x\nmax_x = x + 1\n", "found 3 times"),
        ("# Blank: f(x)\n", "followed by a line of code"),
        ("# Blank: f(x)\n\ny = f(x)\n", "followed by a line of code"),
        ("# Blank: f(x)\n# Hide: hard\ny = f(x)\n", "followed by a line of code"),
        ("# Blank:\ny = f(x)\n", "nothing to blank"),
        ("# Hide: hard\n# Blank[hard]: f(x)\ny = f(x)\n", "hidden in every notebook"),
        ("# Blank: f(x)\n# Blank: f(x)\ny = f(x)\n", "after the other blanks"),
    ],
)
def test_blank_errors(source, error):
    with pytest.raises(ValueError, match=error):
        generate_exercise_notebooks(make_notebook(source))


def test_sync_reports_directive_error_with_file(tmp_path, capsys):
    path = tmp_path / "workshop.ipynb"
    notebook = make_notebook("# Blank: g(x)\ny = f(x)\n")
    path.write_text(notebook_to_str(notebook), encoding="utf-8")
    with pytest.raises(typer.Exit):
        sync([path])
    assert f"{path}: 'g(x)' found 0 times" in capsys.readouterr().err
