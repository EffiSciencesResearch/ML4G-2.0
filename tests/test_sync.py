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

    for output, expected in [(output_normal, expected_normal), (output_hard, expected_hard)]:
        for line1, line2 in zip(output.read_text().splitlines(), expected.read_text().splitlines()):
            assert line1 == line2
        assert len(output.read_text()) == len(expected.read_text())
