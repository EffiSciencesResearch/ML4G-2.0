#!/usr/bin/env python

from copy import deepcopy
from dataclasses import dataclass
import difflib
import json
import re
from pathlib import Path
from subprocess import check_output
import sys
from typing import Annotated, Iterator

import typer
from rich import print as rprint
import black


app = typer.Typer(no_args_is_help=True, add_completion=False)

ROOT = Path(__file__).resolve().parent.parent

RE_URL = re.compile(r"(https?://[^\s)\"]+)([\s).,\\\n]|$)")
BAD_URL_PATTERNS = ["/ML4G/"]

BADGE_TEMPLATE = """<a href="https://colab.research.google.com/github/EffiSciencesResearch/{repo_path}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>"""
RE_BADGE = re.compile(BADGE_TEMPLATE.format(repo_path=r'[^"]+'), re.MULTILINE)
RE_BADGE = re.compile(r'<a href=\\"([^"]+)\\"[^>]*>.*?colab-badge\.svg.*?</a>', re.MULTILINE)

type Notebook = dict


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
    file.write_text(notebook_to_str(notebook), encoding='utf-8')


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
    """

    exercise_notebooks = {}

    def parse_hide(line: str) -> set[str]:
        if line.strip().startswith("# Hide:"):
            return {label.strip().lower() for label in line.split(":")[1].split(",")}
        return set()

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

    # Find labels
    labels = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for line in cell["source"]:
                labels = labels.union(parse_hide(line))

    if not labels:
        return {}

    labels.discard("none")
    labels.discard("solution")
    labels.discard("all")
    labels.add("normal")

    # Generate notebooks
    for label in labels:
        new_notebook = deepcopy(notebook)
        new_cells = []

        solution_lines = []

        for cell in new_notebook["cells"]:
            if cell["cell_type"] == "code":
                hide = False
                hide_in_solution = False
                any_hidden = False
                lines_hidden_in_a_row = []
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
                        lines_hidden_in_a_row.append(line)

                    if not hide and not hides_defined_here:
                        add_line_count_if_needed(lines_hidden_in_a_row)
                        lines_hidden_in_a_row = []
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

            if solution_lines:
                new_cells.append(solution_lines_to_cell(solution_lines))
                solution_lines = []

        new_notebook["cells"] = new_cells
        exercise_notebooks[label] = new_notebook

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


# ------------------------------
# CLI commands
# ------------------------------


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
            file.write_text(notebook_to_str(notebook), encoding='utf-8')


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
            file.write_text(notebook_to_str(with_badge), encoding='utf-8')
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
                out_path.write_text(notebook_to_str(new_notebook), encoding='utf-8')
                print(f"📝 {out_path} generated")


@app.command()
def clean(files: list[Path]):
    """Clean the output and metadata of the notebooks."""

    for file in gather_ipynbs(files):
        notebook = load_notebook(file)
        notebook = clean_notebook(notebook)
        file.write_text(notebook_to_str(notebook), encoding='utf-8')


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

    readme.write_text(content, encoding='utf-8')


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

    file.write_text(json.dumps(notebook, indent=4) + "\n", encoding='utf-8')
    print("✅ Saved!")


@app.command(no_args_is_help=True)
def one_on_ones(
    names: Annotated[str, typer.Argument(help="Space or newline separated list of participants.")],
    rounds: int,
    tries: int = 100,
    preferences: str = None,
):
    """Generate a schedule for one-on-ones for the given names.

    The names are paired in a round-robin fashion, but keeping preferences into account.



    Examples:

        python meta/tools.py one-on-ones "Alice Bob Charlie" 2 --preferences "Alice,Charlie Bob,Alice"

        python meta/tools.py one-on-ones "$(cat names.txt)" 5 --preferences "$(cat preferences.txt)"

    Where names.txt contains the list of space or newline separated names,
    and preferences.txt contains the list of comma separated preferences.
    Each preference is of the form "person1,person2" and means that person1 prefers to be paired with person2.
    """

    names = names.strip()
    if "\n" in names:
        name_list = names.splitlines()
    else:
        name_list = names.split()

    if preferences is not None:
        if "\n" in preferences:
            preferences_list = [line.split(",") for line in preferences.splitlines()]
        else:
            preferences_list = [line.split(",") for line in preferences.split()]

        preferences_dict = {tuple(pref): 2 for pref in preferences_list}

    for i in range(tries):
        try:
            all_pairings = make_pairing_graph(name_list, rounds, preferences_dict)
            print(f"Found a solution after {i+1} tries.", file=sys.stderr)
            break
        except KeyError:
            continue
    else:
        print(f"Failed to find a solution after {tries} tries. Retry?", file=sys.stderr)
        exit(1)

    for person, matches in all_pairings.items():
        print(person, *matches, sep=",")


def make_pairing_graph(
    # names: list[str], rounds: int, forced_pairs: list[tuple[str, str]]
    names: list[str],
    rounds: int,
    preferences: dict[tuple[str, str], int],
    ta_group: set[str],
    ta_group_weight: int = 2,
) -> dict[str, list[str]]:
    """
    Generate a pairing schedule for a given list of names over a specified number of rounds,
    taking into account preferences and a special group of names (TA group).

    Args:
        names (list[str]): A list of unique names to be paired.
        rounds (int): The number of rounds for which the pairings should be generated.
        preferences (dict[tuple[str, str], int]): A dictionary where keys are tuples of names
            representing pairs, and values are integers representing the weight of the preference
            for that pair. A weight of 0 means the pair should not be matched.
        ta_group (set[str]): A set of names that belong to a special group (TA group).
            These names have special pairing rules.
        ta_group_weight (int, optional): The weight to be assigned to edges between names in the
            TA group and other names. Defaults to 2.

    Returns:
        dict[str, list[str]]: A dictionary where keys are names and values are lists of names
            representing the pairings for each round. Each list has a length equal to the number
            of rounds, and the ith element in the list is the name of the person paired with the
            key name in the ith round.

    Raises:
        AssertionError: If any of the following conditions are not met:
            - All names in the preference pairs are in the names list.
            - All names in the names list are unique.
            - All preference pairs are of length 2.
            - All names in the TA group are in the names list.

    Special Rules:
    - TA Group: Names in the `ta_group` have special pairing rules.
        People outside of the TA group will meet with a member of the group only when everyone
        else has met with the group as many times as them.
        Members of the TA group cannot meet between themselves.
    - If the number of names is odd, a dummy name "-" is added to the list of names.
        And we rotate who is not paired with anyone.
    """

    assert all(name in names for pair in preferences.keys() for name in pair)
    assert len(names) == len(set(names)), "Names must be unique"
    assert all(len(pair) == 2 for pair in preferences.keys()), "Preference pairs must be pairs"
    assert all(name in names for name in ta_group), "TA names must be in the names list"

    import networkx as nx

    meets_from_group = {name: 0 for name in names}
    if len(names) % 2 == 1:
        names.append("-")

    G = nx.Graph()
    G.add_nodes_from(names)
    # We make a complete graph
    G.add_edges_from((name, match) for name in names for match in names if name != match)
    # We remove or update edges based on preferences
    for (a, b), weight in preferences.items():
        if weight == 0:
            G.remove_edge(a, b)
        else:
            G[a][b]["weight"] = weight
    # Remove edges between members of the TA group
    G.remove_edges_from((a, b) for a in ta_group for b in ta_group if a != b)

    all_pairings: dict[str, list[str]] = {name: ["-"] * rounds for name in names}

    for round in range(rounds):
        # Adapt the weight of edges between non-meet group members to meetgroup
        # They should meet with the group iff everyone else as met as much as them with the group
        max_meet_number = max(meets_from_group.values())
        for name in names:
            if name in ta_group:
                continue

            for meet in ta_group:
                if name != "-" and G.has_edge(name, meet):
                    default_edge_weight = preferences.get(
                        (name, meet), preferences.get((meet, name))
                    )
                    if default_edge_weight is None:
                        G[name][meet]["weight"] = (
                            ta_group_weight if meets_from_group[name] < max_meet_number else 1
                        )

        # Get a maximum matching
        matching: set[tuple[str, str]] = nx.max_weight_matching(G)
        # Remove the edges in the matching
        G.remove_edges_from(matching)
        # Add the matching to the pairings
        for a, b in matching:
            all_pairings[a][round] = b
            all_pairings[b][round] = a

            # Update the number of meetings with the group
            if a in ta_group:
                meets_from_group[b] += 1
            elif b in ta_group:
                meets_from_group[a] += 1

    all_pairings.pop("-", None)
    return all_pairings


if __name__ == "__main__":
    app()
