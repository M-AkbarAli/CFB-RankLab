"""
Main Streamlit application for CFP Predictor.

This is the entry point for the web application.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from .utils import (
    load_current_season_data,
    get_current_week,
    run_simulation
)
from .components.game_selector import game_selector, conference_championship_selector
from .components.rankings import display_rankings, display_rankings_comparison
from .components.bracket import display_bracket, display_playoff_summary

# Page configuration
st.set_page_config(
    page_title="CFP Predictor",
    page_icon="🏈",
    layout="wide"
)

# Title
st.title("🏈 College Football Playoff Predictor")
st.markdown(
    "Simulate 'what-if' scenarios and see how they affect CFP rankings and playoff selection."
)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Season selector
    current_year = 2024  # Could be dynamic
    season = st.number_input(
        "Season",
        min_value=2014,
        max_value=current_year,
        value=current_year,
        step=1
    )
    
    # Model path
    model_path = Path(__file__).parent.parent.parent / "data" / "models" / "cfp_predictor.pkl"
    
    if not model_path.exists():
        st.error("⚠️ Model not found. Please train the model first.")
        st.stop()
    
    st.caption(f"Using model: {model_path.name}")

# Load data
@st.cache_data
def load_data(season: int):
    """Load season data with caching."""
    return load_current_season_data(season)

try:
    data = load_data(season)
    games_df = data["games"]
    teams_df = data["teams"]
    rankings_df = data["rankings"]
    champions_df = data["champions"]
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Determine current week
current_week = get_current_week(season, games_df)

# Main content
st.header("Scenario Simulation")

# Tabs
tab1, tab2, tab3 = st.tabs(["Game Selector", "Projected Rankings", "Playoff Bracket"])

with tab1:
    st.subheader("Select Game Outcomes")
    
    # Get user-selected outcomes
    selected_outcomes = game_selector(
        games_df=games_df,
        current_week=current_week,
        season=season
    )
    
    # Conference championship selector (if applicable)
    if current_week >= 14:  # Championship week
        st.write("---")
        conf_champions = conference_championship_selector(
            games_df=games_df,
            teams_df=teams_df,
            season=season,
            week=15
        )

with tab2:
    st.subheader("Projected CFP Rankings")
    
    if st.button("Run Projection", type="primary"):
        if not selected_outcomes:
            st.warning("Please select at least one game outcome.")
        else:
            with st.spinner("Computing projections..."):
                # Run simulation
                results = run_simulation(
                    games_df=games_df,
                    teams_df=teams_df,
                    game_outcomes=selected_outcomes,
                    season=season,
                    target_week=15,  # Final week
                    model_path=model_path,
                    champions_df=champions_df
                )
                
                # Store in session state
                st.session_state["simulation_results"] = results
                
                # Display rankings
                if not results["rankings"].empty:
                    display_rankings(
                        rankings_df=results["rankings"],
                        title="Projected Top 25",
                        show_features=True,
                        highlight_playoff=True,
                        playoff_teams=set(results["playoff_teams"]["team"].values) if not results["playoff_teams"].empty else None
                    )
                else:
                    st.error("Failed to generate rankings.")
    
    # Display stored results if available
    if "simulation_results" in st.session_state:
        results = st.session_state["simulation_results"]
        if not results["rankings"].empty:
            display_rankings(
                rankings_df=results["rankings"],
                title="Projected Top 25",
                show_features=True,
                highlight_playoff=True,
                playoff_teams=set(results["playoff_teams"]["team"].values) if not results["playoff_teams"].empty else None
            )

with tab3:
    st.subheader("Projected Playoff Bracket")
    
    if "simulation_results" in st.session_state:
        results = st.session_state["simulation_results"]
        
        if not results["playoff_teams"].empty:
            display_bracket(
                playoff_teams=results["playoff_teams"],
                matchups_df=results["matchups"],
                title="12-Team CFP Playoff Bracket"
            )
            
            st.write("---")
            
            display_playoff_summary(
                playoff_teams=results["playoff_teams"],
                rankings_df=results["rankings"]
            )
        else:
            st.warning("No playoff teams generated. Run a simulation first.")
    else:
        st.info("Run a simulation in the 'Projected Rankings' tab to see the playoff bracket.")

# Footer
st.write("---")
st.caption(
    "This tool uses a machine learning model trained on historical CFP committee rankings. "
    "Results are projections and not official committee decisions."
)

