"""
Main Streamlit application for CFP Predictor.

This is the entry point for the web application.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv(project_root / ".env")

import streamlit as st
import pandas as pd
from typing import Dict, Optional

from src.app.utils import (
    load_current_season_data,
    get_current_week,
    run_simulation
)
from src.app.components.game_selector import game_selector, conference_championship_selector
from src.app.components.rankings import (
    display_rankings, 
    display_rankings_comparison,
    display_team_resume,
    display_team_comparison,
    display_feature_importance
)
from src.app.components.bracket import display_bracket, display_playoff_summary

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

# Mode selector
simulation_mode = st.radio(
    "Simulation Mode",
    options=["Final Rankings", "Weekly Rankings"],
    horizontal=True,
    help="Final Rankings: Predict Selection Day outcome. Weekly Rankings: See ranking evolution week-by-week."
)

# Tabs
if simulation_mode == "Final Rankings":
    tab1, tab2, tab3, tab4 = st.tabs(["Game Selector", "Projected Rankings", "Playoff Bracket", "Team Analysis"])
else:
    tab1, tab2, tab3, tab4 = st.tabs(["Game Selector", "Weekly Rankings", "Playoff Bracket", "Team Analysis"])

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
    if simulation_mode == "Final Rankings":
        st.subheader("Projected CFP Rankings")
        
        if st.button("Run Projection", type="primary"):
            if not selected_outcomes:
                st.warning("Please select at least one game outcome.")
            else:
                with st.spinner("Computing projections..."):
                    # Get previous week's rankings for feature computation
                    prev_week_rankings = None
                    if not rankings_df.empty and current_week > 1:
                        prev_week_rankings = rankings_df[
                            (rankings_df["season"] == season) &
                            (rankings_df["week"] == current_week - 1)
                        ]
                    
                    # Run simulation
                    results = run_simulation(
                        games_df=games_df,
                        teams_df=teams_df,
                        game_outcomes=selected_outcomes,
                        season=season,
                        target_week=15,  # Final week
                        model_path=model_path,
                        champions_df=champions_df,
                        previous_rankings_df=prev_week_rankings
                    )
                    
                    # Store in session state
                    st.session_state["simulation_results"] = results
                    st.session_state["simulation_features"] = None  # Will be computed if needed
                    
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
    
    else:  # Weekly Rankings mode
        st.subheader("Weekly Ranking Evolution")
        
        if st.button("Run Weekly Projection", type="primary"):
            if not selected_outcomes:
                st.warning("Please select at least one game outcome.")
            else:
                with st.spinner("Computing weekly projections..."):
                    from ..simulation.engine import SimulationEngine
                    from ..features.compute import compute_features
                    
                    engine = SimulationEngine(model_path=model_path)
                    
                    # Get start and end weeks
                    start_week = st.session_state.get("start_week", current_week + 1)
                    end_week = st.session_state.get("end_week", 15)
                    
                    weekly_results = engine.simulate_weekly_rankings(
                        base_games_df=games_df,
                        base_teams_df=teams_df,
                        game_outcomes=selected_outcomes,
                        start_week=start_week,
                        end_week=end_week,
                        season=season,
                        base_rankings_df=rankings_df,
                        champions_df=champions_df
                    )
                    
                    st.session_state["weekly_results"] = weekly_results
                    
                    # Display weekly rankings
                    for week, week_rankings in sorted(weekly_results.items()):
                        with st.expander(f"Week {week} Rankings"):
                            display_rankings(
                                rankings_df=week_rankings,
                                title=f"Week {week} Top 25",
                                show_features=False
                            )
        
        # Display stored weekly results
        if "weekly_results" in st.session_state:
            weekly_results = st.session_state["weekly_results"]
            for week, week_rankings in sorted(weekly_results.items()):
                with st.expander(f"Week {week} Rankings"):
                    display_rankings(
                        rankings_df=week_rankings,
                        title=f"Week {week} Top 25",
                        show_features=False
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

with tab4:
    st.subheader("Team Analysis & Feature Transparency")
    
    # Feature importance
    if st.checkbox("Show Feature Importance"):
        try:
            from ..model.predict import load_model
            model = load_model(model_path)
            if isinstance(model, dict):
                model = model.get("model", model)
            
            # Get feature names from model or use defaults
            if hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_
            else:
                # Default feature list
                feature_names = [
                    "wins", "losses", "win_pct", "sos_score", "weighted_sos_score",
                    "wins_vs_winning_teams", "wins_vs_top25", "record_strength_score",
                    "head_to_head_wins_vs_ranked", "is_power5", "is_conference_champion"
                ]
            
            display_feature_importance(model, feature_names)
        except Exception as e:
            st.warning(f"Could not load feature importance: {e}")
    
    # Team resume view
    st.write("---")
    st.write("**View Team Resume**")
    
    if "simulation_results" in st.session_state:
        results = st.session_state["simulation_results"]
        if not results["rankings"].empty:
            team_list = results["rankings"]["team"].tolist()
            selected_team = st.selectbox("Select Team", team_list)
            
            if selected_team:
                # Compute features for this team (would need to recompute or store)
                st.info("Team resume view requires feature recomputation. Feature coming soon.")
                # display_team_resume(selected_team, features_df)
    
    # Team comparison
    st.write("---")
    st.write("**Compare Two Teams**")
    
    if "simulation_results" in st.session_state:
        results = st.session_state["simulation_results"]
        if not results["rankings"].empty:
            team_list = results["rankings"]["team"].tolist()
            col1, col2 = st.columns(2)
            
            with col1:
                team1 = st.selectbox("Team 1", team_list, key="team1")
            with col2:
                team2 = st.selectbox("Team 2", team_list, key="team2")
            
            if team1 and team2 and team1 != team2:
                st.info("Team comparison requires feature recomputation. Feature coming soon.")
                # display_team_comparison(team1, team2, features_df)

# Footer
st.write("---")
st.caption(
    "This tool uses a machine learning model trained on historical CFP committee rankings. "
    "Results are projections and not official committee decisions."
)

