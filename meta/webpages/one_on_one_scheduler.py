from utils.streamlit_utils import State
from tools import make_pairing_graph

import streamlit as st
import pandas as pd
import csv
from io import StringIO


state = State()
with st.sidebar:
    camp = state.login_form()


st.title("One-on-One Scheduler")
st.write(
    """
This tool helps schedule 1-1s between participants, keeping preferences into account.
Rules:
- Two person cannot meet more than once.
- If "Handle TAs separately" is checked
    - Participants will meet with TAs only when all other participants have met with a TA as much as them
    - TAs will not be paired with each other
- Preferences can be added to ensure a pair can meet early. However if TAs are handled separately,
"""
)

# Input section
if camp:
    names_str = "\n".join([name.split()[0] for name in camp.participants_list()])
else:
    names_str = "Alice\nBob\nCharlie\nDavid\nEve\nFrank\nGrace\nHeidi\nIvan\nJudy"


if st.checkbox("Handle TAs separately, and ensure fair repartition of TA time", value=True):
    participant_col, ta_col = st.columns(2)
    with ta_col:
        ta_input = st.text_area(
            "Teaching Assistants (one per line)",
            help="Enter the names of teaching assistants, one per line or separated by spaces",
            height=300,
        )
        tas = [name for name in ta_input.splitlines() if name.strip()]
else:
    participant_col = st.columns(1)[0]
    tas = []

with participant_col:
    names_input = st.text_area(
        "Participants (one per line)",
        value=names_str,
        height=300,
        help="Enter the names of participants, one per line or separated by spaces",
    )
    names = [name for name in names_input.splitlines() if name.strip()]

rounds = st.slider(
    "Number of rounds of 1-1s", min_value=1, max_value=len(names) - 1, value=len(names) * 2 // 3
)
tries = st.number_input(
    "Maximum tries",
    min_value=1,
    max_value=1000,
    value=1000,
    help="Number of attempts to find a valid schedule",
)

prefer_col, against_col = st.columns(2)

with prefer_col:
    preferences_text = st.text_area(
        'Optional Preferences (pairs "name1,name2")',
        help="Format: person1,person2 (one per line). Each preference means person1 prefers to be paired with person2",
        height=300,
    )

with against_col:
    against_text = st.text_area(
        'Pairs not to match (pairs "name1,name2")',
        help="Format: person1,person2 (one per line). Each pair will not be matched",
        height=300,
    )

names = names + tas


# Parse preferences
def is_valid_pair_line(line: str) -> tuple[str, str] | tuple[None, None]:
    pair = line.strip()
    if not pair:
        return None, None

    if len(pair.split(",")) != 2:
        st.error(
            f"Invalid pair format: {pair}. Each pair should be in the format 'person1,person2'."
        )
        st.stop()

    p1, p2 = pair.split(",")
    if p1 not in names or p2 not in names:
        st.error(f"Pair {p1},{p2} contains names not in the participants list.")
        st.stop()

    return p1, p2


preferences = {}
for pref in preferences_text.splitlines():
    p1, p2 = is_valid_pair_line(pref)
    if p1 is not None:
        preferences[p1, p2] = 2

for against in against_text.splitlines():
    p1, p2 = is_valid_pair_line(against)
    if p1 is not None:
        preferences[p1, p2] = 0

# Processing and results display
if st.button("Try to generate schedule"):

    if len(names) < 2:
        st.error("Please enter at least 2 participants.")
        st.stop()

    if len(names) % 2 != 0:
        st.warning(
            "The number of participants is odd. One person will not be paired in each round."
        )

    # Try to generate the schedule

    with st.spinner(f"Generating schedule (up to {tries} attempts)..."):
        for i in range(tries):
            try:
                all_pairings = make_pairing_graph(names, rounds, preferences, set(tas))
                st.success(f"Found a solution after {i+1} attempts!")
                break
            except KeyError:
                continue
        else:
            st.error(
                f"Failed to find a solution after {tries} attempts. Try increasing the number of attempts or adjusting preferences."
            )
            st.stop()

    # Display results if successful
    # Create a DataFrame for display
    df = pd.DataFrame.from_dict(all_pairings, orient="index")
    df.columns = [f"Round {i+1}" for i in range(rounds)]

    st.subheader("Schedule")
    st.dataframe(df)

    # Create CSV for download
    csv_buffer = StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["Participant"] + [f"Round {i+1}" for i in range(rounds)])
    for person, matches in all_pairings.items():
        writer.writerow([person] + matches)

    # Write it in a code block
    st.write("CSV easy to copy and paste to gsheets:")
    st.code(csv_buffer.getvalue(), language="csv")
