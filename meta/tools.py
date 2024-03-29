#!/bin/env python

from dataclasses import dataclass
import json
import re
from pathlib import Path
from subprocess import check_output
from typing import Generator, Iterator

import typer
from rich import print as rprint

app = typer.Typer()

ROOT = Path(__file__).resolve().parent.parent

RE_URL = re.compile(r"(https?://[^\s)\"]+)([\s).,\\\n]|$)")


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


@app.command()
def badge(files: list[Path]):
    """Print code of the "open in colab" badge every input file. Scans directories recursively."""

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
        if "colab-badge.svg" in content:
            print(f"✅ {file} already has a badge")
            continue

        relative_path = file.resolve().relative_to(ROOT)
        badge_content = f"""<a href="https://colab.research.google.com/github/EffiSciencesResearch/ML4G-2.0/blob/master/{relative_path}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>"""

        # Add the badge to the first markdown cell
        parsed = json.loads(content)
        for cell in parsed["cells"]:
            if cell["cell_type"] == "markdown":
                cell["source"].insert(0, badge_content)
                break

        file.write_text(json.dumps(parsed, indent=2))
        print(f"🖊  {file} now has a badge!")


if __name__ == "__main__":
    app()
