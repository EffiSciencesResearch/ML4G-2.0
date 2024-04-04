#!/bin/env python

from copy import deepcopy
from dataclasses import dataclass
import json
import sys
import re
from pathlib import Path
from subprocess import check_output
from typing import Iterator

import typer
from rich import print as rprint

app = typer.Typer()

ROOT = Path(__file__).resolve().parent.parent

RE_URL = re.compile(r"(https?://[^\s)\"]+)([\s).,\\\n]|$)")


def gather_ipynbs(files: list[Path]) -> list[Path]:
    """Recursively gather all the notebooks in the given directories and files.

    Raises an error if a file does not exist or is not a notebook.
    Meant for command line arguments.
    """

    ipynbs = []
    for file in files:
        if file.is_dir():
            ipynbs.extend(file.rglob("*.ipynb"))
        elif file.exists():
            assert file.suffix == ".ipynb", f"{file} is not a notebook"
            ipynbs.append(file)
        else:
            raise FileNotFoundError(file)

    return ipynbs


@dataclass
class Link:
    file: Path
    line_no: int
    url: str

    def __str__(self):
        return f"{self.file}:{self.line_no} {self.url}"


def get_all_links(verbose: bool = False) -> Iterator[Link]:
    """Yields all the links in the git repository."""

    # Get the files tracked by git
    files = check_output(["git", "ls-files"]).decode().split("\n")

    for file in files:
        if not file:
            continue

        file = Path(file)
        if file.suffix not in [".md", ".py", ".ipynb"]:
            if verbose:
                rprint(f"🙈 [yellow]Skipped {file}[/]")
            continue

        for line_no, line in enumerate(file.read_text().splitlines()):
            for match in RE_URL.finditer(line):
                url = match.group(1)
                for ending in [",", ".", '"', "\\n", '\\"']:
                    if url.endswith(ending):
                        url = url[: -len(ending)]

                yield Link(file, line_no, url)


@app.command()
def show_links(with_file: bool = False, verbose: bool = False):
    """Show all the links in the git repository."""

    for link in get_all_links(verbose):
        if with_file:
            print(link)
        else:
            print(link.url)


BAD_PATTERNS = ["/ML4G/"]


def check_link(link: Link, curl: bool = False) -> str | None:
    """Check if the link is broken."""

    url = link.url

    # If a colab link to this repo, check that the file exists
    colab_link = "https://colab.research.google.com/github/"
    this_repo = colab_link + "EffiSciencesResearch/ML4G-2.0/blob/master/"
    if url.startswith(colab_link):
        if not url.startswith(this_repo):
            return "Colab link to another repo"
        path = Path(url[len(this_repo) :])
        if not path.exists():
            return f"Colab link to non-existing file: {path}"

    for pattern in BAD_PATTERNS:
        if pattern in url:
            return f"Bad pattern: {pattern}"

    if curl:
        try:
            check_output(["curl", "--globoff", "--head", "--silent", link.url])
        except Exception:
            return f"curl failed"

    return None


@app.command()
def check_links(curl: bool = False):
    """Check for broken links."""

    for link in get_all_links():
        error = check_link(link, curl)
        if error is not None:
            print(f"🔴 {link} - {error}")


BADGE_TEMPLATE = """<a href="https://colab.research.google.com/github/EffiSciencesResearch/{repo_path}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>"""
RE_BADGE = re.compile(BADGE_TEMPLATE.format(repo_path=r'[^"]+'), re.MULTILINE)
RE_BADGE = re.compile(r'<a href=\\"([^"]+)\\"[^>]*>.*?colab-badge\.svg.*?</a>', re.MULTILINE)


@app.command()
def badge(files: list[Path]):
    """Add the "Open in Colab" badge to every of input notebook.

    Scans directories recursively. If a badge is already present, it will be updated to the current file path.
    """

    for file in files[:]:
        if file.is_dir():
            files.remove(file)
            files.extend(file.rglob("*.ipynb"))

    for file in files:
        if not file.exists():
            print(f"🙈 {file} does not exist")
            continue
        elif file.suffix != ".ipynb":
            print(f"🚷 {file} is not a notebook")
            continue

        content = file.read_text()
        initial_content = content

        relative_path = file.resolve().relative_to(ROOT)
        badge_content = BADGE_TEMPLATE.format(
            repo_path=f"ML4G-2.0/blob/master/{relative_path.as_posix()}"
        )
        badge_content_escaped = badge_content.replace('"', '\\"')

        # Sync the badge and file name
        content = RE_BADGE.sub(badge_content_escaped, content)

        # Add the badge in the first cell if it is not present
        if badge_content_escaped not in content:
            parsed = json.loads(content)
            for cell in parsed["cells"]:
                if cell["cell_type"] == "markdown":
                    cell["source"].insert(0, badge_content + "\n")
                    break
            content = json.dumps(parsed, indent=2)

        bagdes_count = content.count("colab-badge.svg")
        if bagdes_count > 1:
            details = f" {bagdes_count} badges found in this file. 🤔"
        else:
            details = ""

        if content == initial_content:
            print(f"✅ {file} already has a badge.{details}")
        else:
            file.write_text(content)
            print(f"🖊  {file} now has a badge!{details}")


@app.command()
def sync(file: Path):
    """Generate the exercises notebook from the solutions notebook.

    All lines after annotations such as "Hide: all" or "Hide: hard" are removed.
    Those annotations should be on their own line and are removed too. Until an annotation "Hide: none".
    Labels after "Hide: <name1>, <name2>" can be arbitrary python names, but "none", "all" and "solution" are special.
    This generates notebooks "basename_suffix.ipynb" for each suffix found in the file.
    It will always generate a "basename_normal.ipynb" notebook.
    Replace hidden stuff with "..."

    Example:

    ```python
    print("Always visible")
    # Hide: hard
    print("Hidden in the hard notebook")
    print("And this one too")
    # Hide: all
    print("Hidden in all notebooks but the solution")
    # Hide: solution
    ...
    # Hide: none
    print("Visible again")
    ```

    Will generate:
    - basename_normal.ipynb: "Always visible", "Hidden in the hard notebook", "And this one too", "...", "Visible again"
    - basename_hard.ipynb: "Always visible", "...", "Visible again"
    And the solution after this cell will contain everything but the "..." line.
    """

    assert file.exists(), f"{file} does not exist"

    notebook = json.loads(file.read_text())

    def parse_hide(line: str) -> set[str]:
        if line.strip().startswith("# Hide:"):
            return {label.strip().lower() for label in line.split(":")[1].split(",")}
        return set()

    # Find labels
    labels = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for line in cell["source"]:
                labels = labels.union(parse_hide(line))

    labels.discard("none")
    labels.discard("solution")
    labels.discard("all")
    labels.add("normal")

    # Generate notebooks
    for label in labels:
        base_file = file.with_stem(file.stem + f"_{label}")

        new_notebook = deepcopy(notebook)
        new_cells = []

        solution_lines = []

        for cell in new_notebook["cells"]:
            if solution_lines:
                new_lines = [
                    "<details>\n",
                    "<summary>Show solution</summary>\n",
                    "\n",
                    "```python\n",
                    *solution_lines,
                    "```\n",
                    "\n",
                    "</details>\n",
                    "\n",
                ]
                if cell["cell_type"] == "markdown":
                    cell["source"] = new_lines + cell["source"]
                else:
                    new_cells.append(
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": new_lines,
                        }
                    )
                solution_lines = []

            if cell["cell_type"] == "code":
                hide = False
                hide_in_solution = False
                any_hidden = False
                new_lines = []
                for line in cell["source"]:
                    hides_defined_here = parse_hide(line)

                    last_line = new_lines[-1] if new_lines else ""
                    last_hide = hide

                    if "solution" in hides_defined_here:
                        hide_in_solution = True
                    elif hides_defined_here:
                        hide_in_solution = False

                    if "all" in hides_defined_here:
                        hide = True
                    elif label in hides_defined_here:
                        hide = True
                    elif hides_defined_here:
                        hide = False

                    # We started hidding stuff, and previous line was not "..."
                    if hide and not last_hide and last_line.strip() != "...":
                        space = line.partition("#")[0]
                        new_lines.append(space + "...\n")

                    if not hide and not hides_defined_here:
                        new_lines.append(line)
                    else:
                        any_hidden = True

                    if not hide_in_solution and not hides_defined_here:
                        solution_lines.append(line)

                cell["source"] = new_lines
                if not any_hidden:
                    solution_lines = []
            new_cells.append(cell)
        new_notebook["cells"] = new_cells

        base_file.write_text(json.dumps(new_notebook, indent=2))
        print(f"📝 {base_file} generated")

    base_file = file.with_name(file.name.replace("-complete", ""))


@app.command()
def clean(files: list[Path]):
    """Clean the output and metadata of the notebooks."""

    for file in gather_ipynbs(files):
        notebook = json.loads(file.read_text())

        notebook["metadata"] = dict(
            language_info=dict(
                name="python",
                pygments_lexer="ipython3",
            )
        )

        for cell in notebook["cells"]:
            if "outputs" in cell:
                cell["outputs"] = []
            if "execution_count" in cell:
                cell["execution_count"] = None
            if "metadata" in cell:
                cell["metadata"] = {}

        file.write_text(json.dumps(notebook, indent=2) + "\n")


@app.command()
def list_of_workshops_readme():
    """Update the list of workshops in the README.md file."""

    def pretty_name(name: str) -> str:
        return name.replace("-", " ").replace("_", " ")

    start = "<!-- start workshops -->"
    end = "<!-- end workshops -->"

    readme = ROOT / "readme.md"
    content = readme.read_text()

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

        notebooks_links = ", ".join(
            f"[{pretty_name(notebook.stem)}](https://colab.research.google.com/github/EffiSciencesResearch/ML4G-2.0/blob/master/{notebook.relative_to(ROOT).as_posix()})"
            for notebook in notebooks
        )

        workshop_name = pretty_name(workshop.name)
        content += f"- {workshop_name}: {notebooks_links}\n"

    content += end

    readme.write_text(content)


if __name__ == "__main__":
    app()
