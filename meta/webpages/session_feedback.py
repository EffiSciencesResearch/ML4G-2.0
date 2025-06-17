import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path

from utils.camp_utils import get_current_camp
from utils.google_utils import SimpleGoogleAPI
from utils.feedback_utils import FeedbackParser
from utils.streamlit_utils import State

# Page config
st.set_page_config(page_title="Session Feedback Analysis", page_icon="📊", layout="wide")

# Authentication
state = State()
with st.sidebar:
    state.login_form()

# Get current camp
camp = get_current_camp()
if not camp:
    st.error("Please select a camp first on the main page.")
    st.stop()

# Initialize API
SERVICE_ACCOUNT_FILE = Path(__file__).parent.parent / "service_account_token.json"
API = SimpleGoogleAPI(SERVICE_ACCOUNT_FILE)

st.title("📊 Session Feedback Analysis")

st.write(f"Analyzing feedback for camp: **{camp.name}**")

# Check if feedback sheet URL is configured
if not camp.feedback_sheet_url:
    st.warning("No feedback sheet URL configured for this camp.")
    st.write("Please configure the feedback sheet URL in the camp settings.")
    if st.button("Go to Camp Settings"):
        st.switch_page("meta/webpages/edit_camp.py")
    st.stop()

# Manual URL override for testing
with st.expander("🔧 Advanced Options", expanded=False):
    st.write("**Temporary URL Override** (for testing or one-time analysis)")
    manual_url = st.text_input(
        "Override feedback sheet URL",
        value="",
        placeholder="https://docs.google.com/spreadsheets/d/...",
        help="Leave empty to use the URL from camp settings",
    )

    show_names = st.checkbox(
        "Show participant names", value=True, help="Uncheck to anonymize feedback"
    )

# Use manual URL if provided, otherwise use camp URL
feedback_url = manual_url.strip() if manual_url.strip() else camp.feedback_sheet_url


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_feedback_data(url: str):
    """Load and parse feedback data from Google Sheets."""
    parser = FeedbackParser(API)
    return parser.parse_feedback_sheet(url)


# Load feedback data
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

    # Display overview statistics
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

    # Session selection
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

    selected_session = st.selectbox(
        "Select a session to analyze:",
        session_names,
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

        if session.entries:
            # Filter and sort entries: text feedback first, then rating-only
            entries_with_text = [
                entry
                for entry in session.entries
                if entry.feedback_text is not None and entry.feedback_text.strip()
            ]
            entries_rating_only = [
                entry
                for entry in session.entries
                if entry.rating is not None
                and (entry.feedback_text is None or not entry.feedback_text.strip())
            ]

            # Show entries with text feedback first
            if entries_with_text:
                st.markdown("**💬 Feedback with Comments:**")
                for entry in entries_with_text:
                    # Compact single-line display
                    rating_text = "⭐" * entry.rating if entry.rating else "📝"
                    name_text = (
                        f"**{entry.participant_name}**"
                        if (show_names and entry.participant_name)
                        else f"*{entry.tab_source}*"
                    )

                    st.write(f"{rating_text} • {name_text}\n\n {entry.feedback_text}")

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
                    for entry in entries:
                        if show_names and entry.participant_name:
                            names.append(entry.participant_name)
                        else:
                            names.append(f"User from {entry.tab_source}")

                    if rating > 0:
                        st.write(f"**{rating}/5**: {', '.join(names)}")
        else:
            st.info("No feedback data available for this session")

    # All sessions comparison
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

except Exception as e:
    st.error(f"Error loading feedback data: {str(e)}")
    st.write("Please check:")
    st.write("- The feedback sheet URL is correct")
    st.write("- The service account has access to the sheet")
    st.write("- The sheet contains the expected column headers")

    with st.expander("Full error details"):
        st.exception(e)
