from copy import deepcopy
from dataclasses import dataclass
import json
import re
from pathlib import Path
from subprocess import check_output
from typing import Iterator

from rich import print as rprint
import black


ROOT = Path(__file__).resolve().parent.parent.parent

RE_URL = re.compile(r"(https?://[^\s)\"]+)([\s).,\\\n]|$)")
BAD_URL_PATTERNS = ["/ML4G/"]

BADGE_TEMPLATE = """<a href="https://colab.research.google.com/github/EffiSciencesResearch/{repo_path}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>"""
RE_BADGE = re.compile(BADGE_TEMPLATE.format(repo_path=r'[^"]+'), re.MULTILINE)
RE_BADGE = re.compile(r'<a href=\\"([^"]+)\\"[^>]*>.*?colab-badge\.svg.*?</a>', re.MULTILINE)

RE_BLANK = re.compile(r"^\s*# Blank(?:\[(?P<labels>[^\]]*)\])?:(?P<text>.*)$")

type Notebook = dict


@dataclass
class Blank:
    """A "# Blank[labels]: text" directive, which blanks text in the line after it."""

    labels: set[str] | None  # None means every exercise notebook
    text: str
    where: str = ""
    applied: bool = False

    def applies_to(self, label: str) -> bool:
        return self.labels is None or "all" in self.labels or label in self.labels


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


def notebook_to_str(notebook: Notebook) -> str:
    """Convert the notebook to a string, suitable for saving to disc."""
    return json.dumps(notebook, indent=4, ensure_ascii=False) + "\n"


def save_notebook(file: Path, notebook: Notebook):
    """Save the notebook to the given file."""
    file.write_text(notebook_to_str(notebook), encoding="utf-8")


def load_notebook(file: Path) -> Notebook:
    """Load the notebook from the given file."""
    return json.loads(file.read_text("utf-8"))


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

        for line_no, line in enumerate(file.read_text("utf-8").splitlines()):
            for match in RE_URL.finditer(line):
                url = match.group(1)
                for ending in [",", ".", '"', "\\n", '\\"']:
                    if url.endswith(ending):
                        url = url[: -len(ending)]

                yield Link(file, line_no, url)


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

    for pattern in BAD_URL_PATTERNS:
        if pattern in url:
            return f"Bad pattern: {pattern}"

    if curl:
        try:
            check_output(["curl", "--globoff", "--head", "--silent", link.url])
        except Exception:
            return "curl failed"

    return None


def add_badge(file: Path, notebook: Notebook) -> Notebook:
    """Return the notebook with the badge added in the first markdown cell."""
    try:
        relative_path = file.resolve().relative_to(ROOT)
    except ValueError:
        # No need to add badges for things outside of the repo.
        # Those are test notebooks.
        return notebook

    if relative_path.parts[0] != "workshops":
        # No badge if not a workshop.
        return notebook

    badge_content = BADGE_TEMPLATE.format(
        repo_path=f"ML4G-2.0/blob/master/{relative_path.as_posix()}"
    )
    badge_content_escaped = badge_content.replace('"', '\\"')

    # Sync the badge and file name
    content = notebook_to_str(notebook)
    content = RE_BADGE.sub(badge_content_escaped, content)
    notebook = json.loads(content)

    # Add the badge in the first cell if it is not present
    if badge_content_escaped not in content:
        for cell in notebook["cells"]:
            if cell["cell_type"] == "markdown":
                cell["source"].insert(0, badge_content + "\n")
                break

    return notebook


def generate_exercise_notebooks(notebook: Notebook) -> dict[str, Notebook]:
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

    "Blank: <text>" blanks part of the next line: in the exercise notebooks where that line
    is visible, the first occurrence of <text> is replaced by "...". The solution keeps the line.
    "Blank[hard]: <text>" or "Blank[hard, normal]: <text>" only blank it in those notebooks.

    ```python
    # Blank[normal]: run_with_cache
    # Blank[hard]: gpt2_small.run_with_cache(tokens)
    logits, cache = gpt2_small.run_with_cache(tokens)
    ```

    Will generate:
    - basename_normal.ipynb: "logits, cache = gpt2_small....(tokens)"
    - basename_hard.ipynb: "logits, cache = ..."
    """

    exercise_notebooks = {}

    def parse_hide(line: str) -> set[str]:
        if line.strip().startswith("# Hide:"):
            return {label.strip().lower() for label in line.split(":")[1].split(",")}
        return set()

    def parse_blank(line: str) -> Blank | None:
        match = RE_BLANK.match(line)
        if match is None:
            return None
        labels = match.group("labels")
        if labels is not None:
            labels = {label.strip().lower() for label in labels.split(",")}
        return Blank(labels, match.group("text").strip())

    def find_blanks(source: list[str], cell_idx: int) -> dict[int, list[Blank]]:
        """Map the index of each blanked line to its blanks, checking they are well-formed."""
        blanks = {}
        pending = []
        for line_idx, line in enumerate(source):
            blank = parse_blank(line)
            if blank is not None:
                blank.where = f"cell {cell_idx}, line {line_idx + 1}: {line.strip()!r}"
                if not blank.text:
                    raise ValueError(f"Blank with nothing to blank, at {blank.where}")
                pending.append(blank)
                continue
            if not pending:
                continue
            if not line.strip() or line.strip().startswith("#"):
                raise ValueError(f"Blank must be followed by a line of code, at {pending[0].where}")
            for blank in pending:
                if blank.text not in line:
                    raise ValueError(
                        f"{blank.text!r} not found in {line.strip()!r}, at {blank.where}"
                    )
            blanks[line_idx] = pending
            pending = []
        if pending:
            raise ValueError(f"Blank must be followed by a line of code, at {pending[0].where}")
        return blanks

    def add_line_count_if_needed(lines_hidden_in_a_row: list[str]):
        nonlocal new_lines  # Unnecessary, but for clarity
        if lines_hidden_in_a_row:
            # We compute the number of words hidden
            nb_words = sum(
                len(re.findall(r"[a-zA-Z0-9_]+", line)) for line in lines_hidden_in_a_row
            )
            unit = "word" if lines_hidden_in_a_row == 1 else "words"
            new_lines[-1] += f"  # TODO: ~{nb_words} {unit}\n"

    def solution_lines_to_cell(solution_lines: list[str]) -> dict | None:
        if not solution_lines:
            return None

        lines = [
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
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines,
        }

    # Find labels and blanks
    labels = set()
    blanks = {}  # (cell index, line index) -> blanks of that line
    for cell_idx, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            for line in cell["source"]:
                labels = labels.union(parse_hide(line))
            for line_idx, line_blanks in find_blanks(cell["source"], cell_idx).items():
                blanks[cell_idx, line_idx] = line_blanks

    if not labels and not blanks:
        return {}

    labels.discard("none")
    labels.discard("solution")
    labels.discard("all")
    labels.add("normal")

    for line_blanks in blanks.values():
        for blank in line_blanks:
            if blank.labels is not None and not blank.labels <= labels | {"all"}:
                unknown = ", ".join(sorted(blank.labels - labels - {"all"}))
                raise ValueError(
                    f"Unknown notebook {unknown} at {blank.where}. Notebooks: {sorted(labels)}"
                )

    # Generate notebooks
    for label in labels:
        new_notebook = deepcopy(notebook)
        new_cells = []

        solution_lines = []

        for cell_idx, cell in enumerate(new_notebook["cells"]):
            if cell["cell_type"] == "code":
                hide = False
                hide_in_solution = False
                any_hidden = False
                lines_hidden_in_a_row = []
                new_lines = []
                for line_idx, line in enumerate(cell["source"]):
                    if parse_blank(line) is not None:
                        continue

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
                        lines_hidden_in_a_row.append(line)

                    if not hide and not hides_defined_here:
                        add_line_count_if_needed(lines_hidden_in_a_row)
                        lines_hidden_in_a_row = []
                        blanked = line
                        for blank in blanks.get((cell_idx, line_idx), []):
                            if blank.applies_to(label):
                                if blank.text not in blanked:
                                    raise ValueError(
                                        f"{blank.text!r} not found in {blanked.strip()!r} "
                                        f"after the other blanks, at {blank.where}"
                                    )
                                blanked = blanked.replace(blank.text, "...", 1)
                                blank.applied = True
                                any_hidden = True
                        new_lines.append(blanked)
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

            if solution_lines:
                new_cells.append(solution_lines_to_cell(solution_lines))
                solution_lines = []

        new_notebook["cells"] = new_cells
        exercise_notebooks[label] = new_notebook

    for line_blanks in blanks.values():
        for blank in line_blanks:
            if not blank.applied:
                raise ValueError(
                    f"Blank is hidden in every notebook it applies to, at {blank.where}"
                )

    return exercise_notebooks


def clean_notebook(notebook: Notebook) -> Notebook:
    notebook = deepcopy(notebook)

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

    return notebook


def fmt_notebook(notebook: Notebook) -> Notebook:
    try:
        return json.loads(
            black.format_ipynb_string(
                notebook_to_str(notebook),
                fast=True,
                mode=black.Mode(line_length=100, is_ipynb=True),
            )
        )
    except black.NothingChanged:
        return notebook


def notebook_matches_file(notebook: Notebook, file: Path) -> bool:
    try:
        return notebook == load_notebook(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return False


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
