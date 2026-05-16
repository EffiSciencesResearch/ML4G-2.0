# one_on_ones

Pairing scheduler for bootcamp one-on-ones: given a list of participants and a number of rounds, produces a schedule where everyone meets a different partner each round while respecting preferences, unavailability, and TA-group balancing.

## Recommended interface

**Use the web UI** — the "One On One Scheduler" page of the internal Streamlit app. It's the supported way to drive this tool: it handles preference editing, unavailability, TA-group selection, and re-rolls when no solution is found.

The CLI here is the substrate the web page calls into, and is exposed for power users who want to script schedules.

## CLI usage

```bash
uv run python -m meta.one_on_ones <names> <rounds> [--preferences ...] [--tries N]
```

Example:

```bash
uv run python -m meta.one_on_ones "Alice Bob Charlie Dave" 3 --preferences "Alice,Bob"
```

`names` is a space- or newline-separated list of participants; `preferences` is a list of `person1,person2` pairs (also space- or newline-separated) indicating a stronger desire to pair those two.

## Algorithm

`make_pairing_graph` builds a weighted graph over participants and runs NetworkX's `max_weight_matching` once per round, removing matched edges between rounds so people don't repeat partners. TA-group balancing is layered on top: non-TAs are matched with TAs at the same rate, and TAs are never paired with each other. Odd rounds (including those caused by unavailability) get a dummy `-` slot that rotates across participants.
