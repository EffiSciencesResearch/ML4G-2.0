#!/usr/bin/env python

from copy import deepcopy
from dataclasses import dataclass
import difflib
import json
import re
from pathlib import Path
from subprocess import check_output
from typing import Iterator

import typer
from rich import print as rprint


app = typer.Typer(no_args_is_help=True, add_completion=False)

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

    def add_line_count_if_needed(lines_hidden_in_a_row: int):
        nonlocal new_lines  # Unnecessary, but for clarity
        if lines_hidden_in_a_row:
            unit = "line" if lines_hidden_in_a_row == 1 else "lines"
            new_lines[-1] += f"  # TODO: ~{lines_hidden_in_a_row} {unit}\n"

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
                    "\n",
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
                lines_hidden_in_a_row = 0
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

                    # We started hidding stuff, and previous line did not start with "..."
                    if hide and not last_hide and not last_line.strip().startswith("..."):
                        space = line.partition("#")[0]
                        new_lines.append(space + "...")  # We add the number of lines hidden

                    # Increment the counter of hiden lines, if non-empty/non-comment line and hidden
                    if (
                        hide
                        and line.strip()
                        and not hide_in_solution
                        and not line.strip().startswith("#")
                    ):
                        lines_hidden_in_a_row += 1

                    if not hide and not hides_defined_here:
                        add_line_count_if_needed(lines_hidden_in_a_row)
                        lines_hidden_in_a_row = 0
                        new_lines.append(line)
                    else:
                        any_hidden = True

                    if not hide_in_solution and not hides_defined_here:
                        solution_lines.append(line)

                # If we end by hidding, lines_hidden_in_a_row is still > 0
                add_line_count_if_needed(lines_hidden_in_a_row)

                cell["source"] = new_lines
                if not any_hidden:
                    solution_lines = []
            new_cells.append(cell)
        new_notebook["cells"] = new_cells

        base_file.write_text(json.dumps(new_notebook, indent=2))
        print(f"📝 {base_file} generated")


@app.command()
def sync_all(folder: Path = Path(".")):
    """Generate the exercises notebooks from the solutions notebooks in the given folder.

    This process is done recursively and only on notebooks that contain # Hide directives.
    """

    for file in folder.rglob("*.ipynb"):
        if "# Hide:" in file.read_text():
            sync(file)


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

        notebooks.sort()
        notebooks_links = ", ".join(
            f"[{pretty_name(notebook.stem)}](https://colab.research.google.com/github/EffiSciencesResearch/ML4G-2.0/blob/master/{notebook.relative_to(ROOT).as_posix()})"
            for notebook in notebooks
        )

        workshop_name = pretty_name(workshop.name)
        content += f"- {workshop_name}: {notebooks_links}\n"

    content += end

    readme.write_text(content)


def fix_typos_lines(lines: list[str]):
    import openai

    if not lines:
        return []

    content = "".join(lines)
    if len(content) > 4000:
        split = len(lines) // 2
        first = fix_typos_lines(lines[:split])
        if not first.endswith("\n"):
            first += "\n"
        return first + fix_typos_lines(lines[split:])

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """Fix the typos and language from the user.
You receive both python code and markdown text.
Keep your edits minimal and keep the same amount of newlines and the start and end.
The text you receive is part of larger files, and your edits are concatenated back.""",
            },
            {"role": "user", "content": content},
        ],
        # We don't want it to wrap stuff in code/md blocks. Those are all the tokens with two backticks.
        # We don't want to remove " `" though, as it is used for inline code.
        logit_bias={
            7559: -2,  # ' ``'
            11592: -2,  #' ``('
            15506: -2,  # '``'
            33153: -2,  # '````'
        },
    )
    return response.choices[0].message.content


def fmt(
    text: str,
    fg: int | tuple[int, int, int] = None,
    bg: int | tuple[int, int, int] = None,
    underline: bool = False,
) -> str:
    """Format the text with the given colors."""

    mods = ""

    if underline:
        mods += "\033[4m"

    if fg is not None:
        if isinstance(fg, int):
            mods += f"\033[38;5;{fg}m"
        else:
            mods += f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m"

    if bg is not None:
        if isinstance(bg, int):
            mods += f"\033[48;5;{bg}m"
        else:
            mods += f"\033[48;2;{bg[0]};{bg[1]};{bg[2]}m"

    if mods:
        text = mods + text + "\033[0m"

    return text


def fmt_diff(diff: list[str]) -> tuple[str, str]:
    """Format the output of difflib.ndiff.

    Returns:
        tuple[str, str]: The two strings (past, new) with the differences highlighted in ANSI colors.
    """

    past = ""
    new = ""
    for line in diff:
        mark = line[0]
        line = line[2:]
        match mark:
            case " ":
                past += line
                new += line
            case "-":
                past += fmt(line, fg=1, underline=True)
            case "+":
                new += fmt(line, fg=2, underline=True)
            case "?":
                pass

    return past, new


@app.command()
def fix_typos(file: Path, code_too: bool = False, select_cells: bool = False):
    """Fix typos in the given file using gpt-3.5.

    Your API key should be in the environment variable OPENAI_API_KEY.
    """

    assert file.suffix == ".ipynb", f"{file} is not a notebook"

    notebook = json.loads(file.read_text())

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

    file.write_text(json.dumps(notebook, indent=2) + "\n")
    print("✅ Saved!")


if __name__ == "__main__":
    app()
