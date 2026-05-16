#!/usr/bin/env python

import difflib
import json
import re
from pathlib import Path

import typer

from meta.notebook_tools.helpers import (
    ROOT,
    add_badge,
    check_link,
    clean_notebook,
    fix_typos_lines,
    fmt,
    fmt_diff,
    fmt_notebook,
    gather_ipynbs,
    generate_exercise_notebooks,
    get_all_links,
    load_notebook,
    notebook_matches_file,
    notebook_to_str,
)


app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def neat(files: list[Path], clean: bool = True, badge: bool = True, fmt: bool = True):
    for file in gather_ipynbs(files):
        initial = load_notebook(file)
        notebook = initial

        if clean:
            notebook = clean_notebook(initial)
            if notebook != initial:
                print(f"🧹 {file} cleaned")

        if badge:
            with_badge = add_badge(file, notebook)
            if with_badge != notebook:
                notebook = with_badge
                print(f"🏷️ {file} now has a badge!")

        if fmt:
            formated = fmt_notebook(notebook)
            if formated != notebook:
                notebook = formated
                print(f"🖊️  {file} was reformated")

        if notebook == initial:
            print(f"🌟 {file} already neat")
        else:
            file.write_text(notebook_to_str(notebook), encoding="utf-8")


@app.command()
def show_links(with_file: bool = False, verbose: bool = False):
    """Show all the links in the git repository."""

    for link in get_all_links(verbose):
        if with_file:
            print(link)
        else:
            print(link.url)


@app.command()
def check_links(curl: bool = False):
    """Check for broken links."""

    for link in get_all_links():
        error = check_link(link, curl)
        if error is not None:
            print(f"🔴 {link} - {error}")


@app.command()
def badge(files: list[Path]):
    """Add the "Open in Colab" badge to every of input notebook.

    Scans directories recursively. If a badge is already present, it will be updated to the current file path.
    """

    for file in gather_ipynbs(files):
        notebook = load_notebook(file)
        with_badge = add_badge(file, notebook)

        bagdes_count = notebook_to_str(with_badge).count("colab-badge.svg")
        if bagdes_count > 1:
            details = f" {bagdes_count} badges found in this file. 🤔"
        else:
            details = ""

        if notebook == with_badge:
            print(f"✅ {file} already has a badge.{details}")
        else:
            file.write_text(notebook_to_str(with_badge), encoding="utf-8")
            print(f"🖊  {file} now has a badge!{details}")


@app.command(help=generate_exercise_notebooks.__doc__)
def sync(files: list[Path]):
    for file in gather_ipynbs(files):
        notebook = load_notebook(file)
        new_notebooks = generate_exercise_notebooks(notebook)

        for label, new_notebook in new_notebooks.items():
            out_path = file.with_stem(file.stem + f"_{label}")
            new_notebook = fmt_notebook(clean_notebook(add_badge(out_path, new_notebook)))

            # Check if there were updates:
            if notebook_matches_file(new_notebook, out_path):
                print(f"✅ {out_path} already up-to-date")
            else:
                out_path.write_text(notebook_to_str(new_notebook), encoding="utf-8")
                print(f"📝 {out_path} generated")


@app.command()
def clean(files: list[Path]):
    """Clean the output and metadata of the notebooks."""

    for file in gather_ipynbs(files):
        notebook = load_notebook(file)
        notebook = clean_notebook(notebook)
        file.write_text(notebook_to_str(notebook), encoding="utf-8")


@app.command()
def list_of_workshops_readme():
    """Update the list of workshops in the README.md file."""

    def pretty_name(name: str) -> str:
        return name.replace("-", " ").replace("_", " ")

    start = "<!-- start workshops -->"
    end = "<!-- end workshops -->"

    readme = ROOT / "readme.md"
    content = readme.read_text("utf-8")

    start_idx = content.index(start) + len(start)
    end_idx = content.index(end)

    end = content[end_idx:]
    content = content[:start_idx] + "\n"

    for workshop in sorted((ROOT / "workshops").iterdir()):
        if not workshop.is_dir():
            continue

        notebooks = list(workshop.glob("*.ipynb"))
        if not notebooks:
            continue

        notebooks.sort()
        notebooks_links = ", ".join(
            f"[{pretty_name(notebook.stem)}](https://colab.research.google.com/github/EffiSciencesResearch/ML4G-2.0/blob/master/{notebook.relative_to(ROOT).as_posix()})"
            for notebook in notebooks
        )

        workshop_name = pretty_name(workshop.name)
        content += f"- {workshop_name}: {notebooks_links}\n"

    content += end

    readme.write_text(content, encoding="utf-8")


@app.command()
def fix_typos(file: Path, code_too: bool = False, select_cells: bool = False):
    """Fix typos in the given file using gpt-3.5.

    Your API key should be in the environment variable OPENAI_API_KEY.
    """

    assert file.suffix == ".ipynb", f"{file} is not a notebook"

    notebook = load_notebook(file)

    if select_cells:
        cells = []
        for idx, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "markdown" and not code_too:
                continue
            content = "".join(cell["source"])
            start = content[:100].strip().replace("\n", "\\n")
            cells.append(f"{idx}: {cell['cell_type']}\t-- {start}...")

        edit = "Keep the cells you want to edit, delete the others:\n\n" + "\n".join(cells)
        selected = typer.edit(edit)
        if selected is None:
            selected = edit

        selected = [int(match) for match in re.findall(r"^(\d+):", selected, re.MULTILINE)]
    elif code_too:
        selected = range(len(notebook["cells"]))
    else:
        selected = [
            idx for idx, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "markdown"
        ]

    for cell_idx in selected:
        cell = notebook["cells"][cell_idx]

        initial = "".join(cell["source"])
        # Gray
        print(fmt(initial, fg=8))

        new = fix_typos_lines(cell["source"])

        if new == initial:
            print("✅ No typos found")
            continue

        # Compute the difference between the two texts
        words1 = re.findall(r"(\w+|\W+)", initial.strip())
        words2 = re.findall(r"(\w+|\W+)", new.strip())

        diff = difflib.ndiff(words1, words2)
        initial_formated, new_formated = fmt_diff(diff)
        print(initial_formated)
        print(new_formated)

        # Ask for confirmation, and allow the user to edit the cell if needed
        while True:
            cmd = input(
                "Accept (enter), edit new (e), edit old (o), skip (s), save and quit (q): "
            ).lower()
            if not cmd or cmd in list("eosq"):
                break

        if cmd == "e":
            edited = typer.edit(new)
            if edited is not None:
                new = edited
        elif cmd == "o":
            edited = typer.edit(initial)
            if edited is not None:
                initial = edited
        elif cmd == "s":
            continue
        elif cmd == "q":
            break

        cell["source"] = new.splitlines(keepends=True)

    file.write_text(json.dumps(notebook, indent=4) + "\n", encoding="utf-8")
    print("✅ Saved!")


if __name__ == "__main__":
    app()
