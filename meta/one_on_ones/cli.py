#!/usr/bin/env python

import sys
from typing import Annotated

import typer

from meta.one_on_ones.pairing import make_pairing_graph


app = typer.Typer(no_args_is_help=True, add_completion=False)


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


if __name__ == "__main__":
    app()
