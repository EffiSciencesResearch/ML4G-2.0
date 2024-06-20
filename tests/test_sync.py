import os
from pathlib import Path
from tools import sync


def test_sync(tmpdir):
    tmpdir = Path(tmpdir)
    base_notebook = Path(__file__).parent / "sample_notebook.ipynb"
    expected_normal = base_notebook.with_name("sample_notebook_normal.ipynb")
    expected_hard = base_notebook.with_name("sample_notebook_hard.ipynb")

    input_notebook = tmpdir / "sample_notebook.ipynb"
    input_notebook.write_text(base_notebook.read_text(), encoding="utf-8")
    output_normal = tmpdir / "sample_notebook_normal.ipynb"
    output_hard = tmpdir / "sample_notebook_hard.ipynb"

    sync(input_notebook)

    assert output_normal.read_text() == expected_normal.read_text()
    assert output_hard.read_text() == expected_hard.read_text()
