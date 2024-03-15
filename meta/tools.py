#!/bin/env python

from dataclasses import dataclass
import re
from pathlib import Path
from subprocess import check_output
from typing import Generator, Iterator

import typer
from rich import print as rprint

app = typer.Typer()

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


BAD_PATTERNS = []


def check_link(link: Link, curl: bool = False) -> bool:
    """Check if the link is broken."""
    if any(pattern in link.url for pattern in BAD_PATTERNS):
        return False

    # If a colab link to this repo, check that the file exists
    colab_link_start = (
        "https://colab.research.google.com/github/EffiSciencesResearch/ML4G/blob/main/"
    )
    if link.url.startswith(colab_link_start):
        path = Path(link.url[len(colab_link_start) :])
        if not path.exists():
            return False

    if curl:
        try:
            check_output(["curl", "--globoff", "--head", "--silent", link.url])
        except Exception:
            return False

    return True


@app.command()
def check_links(curl: bool = False):
    """Check for broken links."""

    for link in get_all_links():
        if not check_link(link, curl):
            print(f"🔴 {link}")


if __name__ == "__main__":
    app()
