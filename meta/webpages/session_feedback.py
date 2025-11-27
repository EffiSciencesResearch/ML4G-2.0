import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

from utils.camp_utils import get_current_camp
from utils.google_utils import SimpleGoogleAPI
from utils.feedback_utils import FeedbackParser
from utils.streamlit_utils import State, render_select_camp_message

# Expected column header patterns
RATING_PATTERN = r"How would you rate the '(.*)'[^']*$"
FEEDBACK_PATTERN = r"Any additional feedback on '(.*)'[^']*$"

# Page config
st.set_page_config(page_title="Session Feedback Analysis", page_icon="📊", layout="wide")


@st.cache_resource
def get_api():
    """Create and cache a SimpleGoogleAPI instance."""
    SERVICE_ACCOUNT_FILE = Path(__file__).parent.parent / "service_account_token.json"
    return SimpleGoogleAPI(SERVICE_ACCOUNT_FILE)


@st.cache_data(ttl=300)  # Cache forever
def load_feedback_data(url: str):
    """Load and parse feedback data from Google Sheets."""
    api = get_api()
    parser = FeedbackParser(api)
    return parser.parse_feedback_sheet(url)


@st.cache_data
def extract_daily_trends(url: str, daily_questions):
    """Extract daily trend data for the three recurring questions."""
    from utils.google_utils import extract_id_from_url

    api = get_api()
    sheet_id = extract_id_from_url(url)
    all_sheets_data = api.get_all_sheets_data(sheet_id)

    daily_data = {}
    day_order = []

    for tab_name, raw_data in all_sheets_data.items():
        if not raw_data or len(raw_data) < 2:
            continue

        headers = raw_data[0]

        # Check if this tab has any of our daily questions
        tab_has_daily_questions = any(
            any(question.lower() in header.lower() for header in headers)
            for question in daily_questions.values()
        )

        if not tab_has_daily_questions:
            continue

        day_order.append(tab_name)
        daily_data[tab_name] = {}

        # Find name column
        name_col_index = -1
        for i, col in enumerate(headers):
            if col.lower().strip() in ["name", "participant name", "full name", "your name"]:
                name_col_index = i
                break

        # Extract data for each daily question
        for metric_name, question_text in daily_questions.items():
            question_col_index = -1
            # Make matching more robust for "Overall Experience"
            search_text = question_text.split(":")[0]  # Use "Overall Experience" for matching

            for i, col in enumerate(headers):
                if search_text.lower() in col.lower():
                    question_col_index = i
                    break

            if question_col_index != -1:
                daily_data[tab_name][metric_name] = []

                for row in raw_data[1:]:
                    if len(row) > question_col_index:
                        rating_str = row[question_col_index]
                        name = (
                            row[name_col_index]
                            if name_col_index != -1 and len(row) > name_col_index
                            else None
                        )

                        try:
                            # Allow 1-10 scale for Energy Levels and Overall Experience, 1-5 for others
                            max_rating = (
                                10 if metric_name in ["Energy Levels", "Overall Experience"] else 5
                            )
                            rating = int(float(rating_str))

                            if 1 <= rating <= max_rating:
                                daily_data[tab_name][metric_name].append(
                                    {"name": name, "rating": rating}
                                )
                        except (ValueError, TypeError):
                            continue

    return daily_data, day_order


def render_overview(feedback_data):
    st.subheader("📈 Overview")
    col1, col2, col3, col4 = st.columns(4)

    total_sessions = len(feedback_data)
    total_responses = sum(len(session.entries) for session in feedback_data.values())
    total_ratings = sum(len(session.ratings) for session in feedback_data.values())
    avg_rating_overall = (
        sum(sum(session.ratings) for session in feedback_data.values()) / total_ratings
        if total_ratings > 0
        else 0
    )

    with col1:
        st.metric("Total Sessions", total_sessions)
    with col2:
        st.metric("Total Responses", total_responses)
    with col3:
        st.metric("Total Ratings", total_ratings)
    with col4:
        st.metric(
            "Overall Average", f"{avg_rating_overall:.1f}/5" if avg_rating_overall > 0 else "N/A"
        )


def render_session_analysis_tab(feedback_data, show_names):
    st.subheader("🎯 Session Analysis")

    session_names = sorted(list(feedback_data.keys()))  # Sort alphabetically

    # Add search functionality
    search_term = st.text_input(
        "🔍 Search sessions:",
        placeholder="Type to filter sessions...",
        help="Search for specific sessions by name",
    )

    # Filter sessions based on search term
    if search_term:
        filtered_sessions = [name for name in session_names if search_term.lower() in name.lower()]
        if not filtered_sessions:
            st.warning(f"No sessions found matching '{search_term}'")
            st.stop()
        session_names = filtered_sessions

    # Always use pills with first item selected by default
    selected_session = st.pills(
        "Select a session to analyze:",
        session_names,
        selection_mode="single",
        default=session_names[0] if session_names else None,
        help="Choose a session to see detailed feedback and ratings",
    )

    if selected_session:
        session = feedback_data[selected_session]

        st.markdown(f"### 📊 {selected_session}")

        # Single compact histogram and metrics
        if session.ratings:
            col1, col2 = st.columns([2, 1])

            with col1:
                fig = px.histogram(
                    x=session.ratings,
                    nbins=5,
                    title=f"Rating Distribution (n={len(session.ratings)})",
                    labels={"x": "Rating", "y": "Count"},
                    color_discrete_sequence=["#1f77b4"],
                    height=300,
                )
                fig.update_xaxes(dtick=1, range=[0.5, 5.5])
                fig.update_traces(marker_line_width=1, marker_line_color="white")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                avg_rating = session.average_rating
                st.metric("Average Rating", f"{avg_rating:.1f}/5" if avg_rating else "N/A")
                st.metric("Total Ratings", len(session.ratings))
                st.metric("Text Feedback", len(session.feedback_texts))
        else:
            st.info("No ratings available for this session")

        # Detailed feedback
        st.markdown("### 💬 Detailed Feedback")

        # Checkbox for session docs format
        format_for_docs = st.checkbox(
            "Format for session docs",
            value=False,
            help="Display feedback in a compact format suitable for copying into session documentation",
        )
        if format_for_docs:
            show_names = False

        if session.entries:
            # Filter and sort entries: text feedback first, then rating-only
            entries_with_text = [
                entry
                for entry in session.entries
                if entry.feedback_text is not None and entry.feedback_text.strip()
            ]
            entries_with_text.sort(key=lambda x: x.rating or 99, reverse=True)
            entries_rating_only = [
                entry
                for entry in session.entries
                if entry.rating is not None
                and (entry.feedback_text is None or not entry.feedback_text.strip())
            ]

            # Show entries with text feedback first
            if entries_with_text:
                st.markdown("**💬 Feedback with Comments:**")
                display_lines = []
                for entry in entries_with_text:
                    rating_text = "⭐" * entry.rating if entry.rating else "📝"
                    if format_for_docs:
                        display_lines.append(f"- {rating_text} • {entry.feedback_text}")
                    else:
                        name_text = (
                            f"**{entry.participant_name}**"
                            if (show_names and entry.participant_name)
                            else "*anonymous*"
                        )
                        display_lines.append(
                            f"{rating_text} • {name_text}\n\n {entry.feedback_text}"
                        )
                if format_for_docs:
                    st.markdown("\n".join(display_lines))
                else:
                    st.write("\n\n".join(display_lines))

            # Show rating-only entries at the end, more compactly
            if entries_rating_only:
                st.markdown("**⭐ Ratings Only:**")

                # Group by rating for compact display
                rating_groups = {}
                for entry in entries_rating_only:
                    rating = entry.rating or 0
                    if rating not in rating_groups:
                        rating_groups[rating] = []
                    rating_groups[rating].append(entry)

                for rating in sorted(rating_groups.keys(), reverse=True):
                    entries = rating_groups[rating]
                    names = []
                    anonymous_count = 0

                    for entry in entries:
                        if show_names and entry.participant_name:
                            names.append(entry.participant_name)
                        else:
                            anonymous_count += 1

                    # Add anonymous count if any
                    if anonymous_count > 0:
                        names.append(f"Anonymous ×{anonymous_count}")

                    if rating > 0:
                        st.write(f"**{rating}/5**: {', '.join(names)}")
            else:
                st.info("No feedback data available for this session")


def render_session_comparison_tab(feedback_data):
    st.subheader("📊 All Sessions Comparison")

    if len(feedback_data) > 1:
        comparison_data = []
        for name, session in feedback_data.items():
            if session.ratings:
                comparison_data.append(
                    {
                        "Session": name,
                        "Average Rating": session.average_rating,
                        "Response Count": len(session.ratings),
                        "Feedback Count": len(session.feedback_texts),
                    }
                )

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)

            # Add sorting option
            sort_by_rating = st.checkbox("Sort by average rating (highest first)", value=False)

            if sort_by_rating:
                comparison_df = comparison_df.sort_values("Average Rating", ascending=False)

            # Bar chart of average ratings
            fig = px.bar(
                comparison_df,
                x="Session",
                y="Average Rating",
                title="Average Ratings by Session",
                color="Average Rating",
                color_continuous_scale="RdYlBu_r",
            )
            fig.update_layout(xaxis_tickangle=-45)
            fig.update_yaxes(range=[1, 5])
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            st.markdown("### 📋 Summary Table")
            st.dataframe(comparison_df.round(1), use_container_width=True, hide_index=True)
        else:
            st.info("No rating data available for comparison")
    else:
        st.info("Need at least 2 sessions for comparison")


def render_daily_trends_tab(daily_data, day_order, daily_questions):
    st.subheader("📈 Daily Trends Analysis")

    try:
        if not daily_data:
            st.info(
                "No daily trend data found. Make sure your sheet has tabs with the three daily questions."
            )
        else:
            # Plot type toggle
            plot_type = st.radio(
                "Visualization type:",
                ["Individual Lines", "Response Bubbles"],
                horizontal=True,
                help="Individual Lines: see each participant's journey. Response Bubbles: see distribution of scores.",
            )

            # Create plots for each metric
            for metric_name in daily_questions.keys():
                st.markdown(f"**{metric_name}**")

                # Determine plot Y-axis range based on metric
                y_axis_range = (
                    [0.5, 10.5]
                    if metric_name in ["Energy Levels", "Overall Experience"]
                    else [0.5, 5.5]
                )

                # Prepare data for plotting
                if plot_type == "Individual Lines":
                    # --- Spaghetti Plot ---
                    fig = go.Figure()

                    # Track all participants and averages
                    participant_data = {}
                    daily_averages = {}

                    for day_idx, day in enumerate(day_order):
                        if day in daily_data and metric_name in daily_data[day]:
                            responses = daily_data[day][metric_name]
                            ratings = [r["rating"] for r in responses]

                            if ratings:
                                daily_averages[day_idx] = sum(ratings) / len(ratings)
                                for response in responses:
                                    # Use a hash for anonymous users to give them a consistent (but meaningless) ID
                                    name = (
                                        response["name"]
                                        or f"Anonymous_{hash(str(response)) % 1000}"
                                    )
                                    if name not in participant_data:
                                        participant_data[name] = {}
                                    participant_data[name][day_idx] = response["rating"]

                    # Assign a color to each participant
                    all_participants = sorted(list(participant_data.keys()))
                    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Alphabet
                    participant_colors = {
                        name: colors[i % len(colors)] for i, name in enumerate(all_participants)
                    }

                    # Add individual participant lines
                    for name, data in participant_data.items():
                        if len(data) >= 2:
                            days = sorted(data.keys())
                            ratings = [data[day] for day in days]
                            fig.add_trace(
                                go.Scatter(
                                    x=days,
                                    y=ratings,
                                    mode="lines",
                                    line=dict(width=1.5, color=participant_colors.get(name)),
                                    name=name,
                                    showlegend=True,
                                    hovertemplate=f"<b>{name}</b><br>Day: %{{x+1}}<br>Rating: %{{y}}/5<extra></extra>",
                                )
                            )

                    # Add average line
                    if daily_averages:
                        avg_days = sorted(daily_averages.keys())
                        avg_ratings = [daily_averages[day] for day in avg_days]
                        fig.add_trace(
                            go.Scatter(
                                x=avg_days,
                                y=avg_ratings,
                                mode="lines+markers",
                                line=dict(width=5, color="rgba(0,0,0,0.8)"),
                                marker=dict(size=8, symbol="diamond"),
                                name="Average",
                                hovertemplate="<b>Average</b><br>Day: %{x+1}<br>Rating: %{y:.1f}/5<extra></extra>",
                            )
                        )

                    fig.update_layout(
                        xaxis_title="Day",
                        yaxis_title="Rating",
                        yaxis=dict(range=y_axis_range, dtick=1),
                        xaxis=dict(
                            tickmode="array",
                            tickvals=list(range(len(day_order))),
                            ticktext=[f"Day {i+1}" for i in range(len(day_order))],
                        ),
                        height=500,
                        showlegend=True,
                        margin=dict(t=20, b=20),
                    )

                else:
                    # --- Bubble Plot ---
                    import math

                    fig = go.Figure()

                    # Fixed colors for ratings
                    rating_colors = {
                        1: "#d73027",
                        2: "#fc8d59",
                        3: "#fee090",
                        4: "#91bfdb",
                        5: "#4575b4",
                        6: "#2166ac",
                        7: "#1a9850",
                        8: "#91cf60",
                        9: "#d9ef8b",
                        10: "#fee08b",  # Added more colors for 1-10
                    }

                    for day_idx, day in enumerate(day_order):
                        if day in daily_data and metric_name in daily_data[day]:
                            responses = daily_data[day][metric_name]
                            rating_counts = {}
                            for response in responses:
                                rating = response["rating"]
                                rating_counts[rating] = rating_counts.get(rating, 0) + 1

                            for rating, count in rating_counts.items():
                                fig.add_trace(
                                    go.Scatter(
                                        x=[day_idx],
                                        y=[rating],
                                        mode="markers",
                                        marker=dict(
                                            size=math.sqrt(count) * 12 + 8,
                                            color=rating_colors.get(rating, "lightgrey"),
                                            line=dict(width=1, color="white"),
                                        ),
                                        name=f"Rating {rating}",
                                        showlegend=False,
                                        hovertemplate=(
                                            f"Day {day_idx+1}<br>Rating: {rating}/10<br>Responses: {count}<extra></extra>"
                                            if metric_name
                                            in ["Energy Levels", "Overall Experience"]
                                            else f"Day {day_idx+1}<br>Rating: {rating}/5<br>Responses: {count}<extra></extra>"
                                        ),
                                    )
                                )

                    fig.update_layout(
                        xaxis_title="Day",
                        yaxis_title="Rating",
                        yaxis=dict(range=y_axis_range, dtick=1),
                        xaxis=dict(
                            tickmode="array",
                            tickvals=list(range(len(day_order))),
                            ticktext=[f"Day {i+1}" for i in range(len(day_order))],
                        ),
                        height=400,
                        margin=dict(t=20, b=20),
                    )

                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error analyzing daily trends: {str(e)}")


def render_day_analysis_tab(daily_data, day_order):
    st.subheader("🔬 Day Analysis")
    if not daily_data:
        st.info("No daily data available for analysis.")
        st.stop()

    selected_day = st.pills("Select a day to analyze", options=day_order, default=day_order[0])

    if selected_day:
        st.write(f"Analyzing data for **{selected_day}**...")

        # --- Linter 1: Missing names ---
        all_names = set()
        for day in day_order:
            if day in daily_data:
                for metric in daily_data[day].values():
                    for response in metric:
                        if response["name"]:
                            all_names.add(response["name"])

        names_on_selected_day = set()
        if selected_day in daily_data:
            for metric in daily_data[selected_day].values():
                for response in metric:
                    if response["name"]:
                        names_on_selected_day.add(response["name"])

        missing_names = sorted(list(all_names - names_on_selected_day))

        if missing_names:
            st.warning(
                f"🕵️ Missing participants ({len(missing_names)}): Participants who submitted feedback on other days but not this one."
            )
            st.table(pd.DataFrame(missing_names, columns=["Name"]))

        # --- Linter 2: Low/Extreme Scores ---
        flagged_scores = []
        if selected_day in daily_data:
            # Create a dict to hold participant scores for easier lookup
            participant_scores = {}
            for metric, responses in daily_data[selected_day].items():
                for response in responses:
                    name = response.get("name")
                    if not name:
                        continue
                    if name not in participant_scores:
                        participant_scores[name] = {}
                    participant_scores[name][metric] = response["rating"]

            for name, scores in participant_scores.items():
                reasons = []
                energy = scores.get("Energy Levels")
                experience = scores.get("Overall Experience")
                content = scores.get("Content Amount")

                if energy is not None and energy <= 4:
                    reasons.append(f"Energy ≤ 4 (is {energy})")
                if experience is not None and experience <= 5:
                    reasons.append(f"Experience ≤ 5 (is {experience})")
                if content is not None and content in [1, 5]:
                    reasons.append(f"Content Amount is {content}")

                if reasons:
                    flagged_scores.append(
                        {
                            "Name": name,
                            "Energy": energy,
                            "Experience": experience,
                            "Content": content,
                            "Reason": ", ".join(reasons),
                        }
                    )

        if flagged_scores:
            st.warning(
                f"🚩 Flagged scores ({len(flagged_scores)}): Participants with low energy/experience, or extreme 'content amount' ratings."
            )
            st.dataframe(pd.DataFrame(flagged_scores), use_container_width=True, hide_index=True)

        # --- Linter 3: Score drop ---
        score_drops = []
        selected_day_index = day_order.index(selected_day)

        if selected_day_index > 0:
            previous_day = day_order[selected_day_index - 1]

            # Get participant scores for previous day
            prev_day_scores = {}
            if previous_day in daily_data:
                for metric, responses in daily_data[previous_day].items():
                    for response in responses:
                        name = response.get("name")
                        if name:
                            if name not in prev_day_scores:
                                prev_day_scores[name] = {}
                            prev_day_scores[name][metric] = response["rating"]

            # Compare with current day
            for name, current_scores in participant_scores.items():
                if name in prev_day_scores:
                    prev_scores = prev_day_scores[name]
                    reasons = []

                    # Check energy drop
                    energy_now = current_scores.get("Energy Levels")
                    energy_prev = prev_scores.get("Energy Levels")
                    if energy_now is not None and energy_prev is not None:
                        if energy_prev - energy_now >= 3:
                            reasons.append(f"Energy drop: {energy_prev} → {energy_now}")

                    # Check experience drop
                    exp_now = current_scores.get("Overall Experience")
                    exp_prev = prev_scores.get("Overall Experience")
                    if exp_now is not None and exp_prev is not None:
                        if exp_prev - exp_now >= 3:
                            reasons.append(f"Experience drop: {exp_prev} → {exp_now}")

                    if reasons:
                        score_drops.append({"Name": name, "Change": ", ".join(reasons)})

        if score_drops:
            st.warning(
                f"📉 Significant score drops ({len(score_drops)}): Participants whose energy or experience dropped by 3+ points from the previous day."
            )
            st.dataframe(pd.DataFrame(score_drops), use_container_width=True, hide_index=True)


def main():
    # Authentication
    state = State()
    with st.sidebar:
        state.login_form()

    st.title("📊 Session Feedback Analysis")
    camp = get_current_camp()
    if not camp:
        render_select_camp_message()
        st.stop()

    st.markdown(
        f"""
    **What this does:** Extracts and analyzes session feedback from Google Sheets containing participant responses across multiple tabs.

    **Input assumptions:**
    - Google Sheets with multiple tabs (each tab processed separately, then aggregated)
    - First row of each tab contains column headers
    - Rating columns must match: `{RATING_PATTERN}`
    - Feedback columns must match: `{FEEDBACK_PATTERN}`
    - Name column (optional): matches patterns like "Name", "Participant Name", etc.
    - Ratings are integers 1-5; text feedback can be any string

    **Output:** Aggregated ratings and feedback for each session, with histograms and individual responses.

    ---
    """
    )

    # Initialize API
    # SERVICE_ACCOUNT_FILE = Path(__file__).parent.parent / "service_account_token.json"
    # API = SimpleGoogleAPI(SERVICE_ACCOUNT_FILE)

    # Check if feedback sheet URL is configured
    if not camp.feedback_sheet_url:
        st.warning("No feedback sheet URL configured for this camp.")
        st.write("Enter the feedback sheet URL below:")

        manual_url = st.text_input(
            "Feedback sheet URL",
            value="",
            placeholder="https://docs.google.com/spreadsheets/d/...",
            help="Google Sheets URL containing session feedback data",
        )

        if not manual_url.strip():
            st.info("Please enter a feedback sheet URL to continue.")
            st.stop()

        feedback_url = manual_url.strip()
    else:
        feedback_url = camp.feedback_sheet_url

    # Show participant names toggle
    show_names = st.checkbox(
        "Show participant names", value=True, help="Uncheck to anonymize feedback"
    )

    # Cache management
    with st.sidebar:
        st.markdown("---")
        if st.button("🗑️ Clear Cache", help="Clear cached feedback data to reload from sheet"):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()

    try:
        with st.spinner("Loading feedback data..."):
            feedback_data = load_feedback_data(feedback_url)

        if not feedback_data:
            st.warning("No feedback data found in the sheet. Please check:")
            st.write(
                """
            - The sheet has the correct column headers (e.g., "How would you rate the 'Project brainstorm' session?")
            - The sheet is accessible to the service account
            - The sheet contains data beyond just headers
            """
            )
            st.stop()

        render_overview(feedback_data)

        # Extract daily metrics from tabs that have the relevant questions
        daily_questions = {
            "Content Amount": "What do you think about the amount of content we had today?",
            "Overall Experience": "Overall Experience: How would you rate your overall experience of the past 24 hours?",
            "Energy Levels": "How are your energy levels?",
        }

        daily_data, day_order = extract_daily_trends(feedback_url, daily_questions)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Session analysis", "Session comparison", "Daily trends", "Day analysis"]
        )

        with tab1:
            render_session_analysis_tab(feedback_data, show_names)
        with tab2:
            render_session_comparison_tab(feedback_data)
        with tab3:
            render_daily_trends_tab(daily_data, day_order, daily_questions)
        with tab4:
            render_day_analysis_tab(daily_data, day_order)

    except Exception as e:
        st.error(f"Error loading feedback data: {str(e)}")
        st.write("Please check:")
        st.write("- The feedback sheet URL is correct")
        st.write("- The service account has access to the sheet")
        st.write("- The sheet contains the expected column headers")

        with st.expander("Full error details"):
            st.exception(e)


main()
