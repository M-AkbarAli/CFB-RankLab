"""
Bracket display component for Streamlit app.
"""

import pandas as pd
import streamlit as st
from typing import Optional, Set
import plotly.graph_objects as go


def display_bracket(
    playoff_teams: pd.DataFrame,
    matchups_df: Optional[pd.DataFrame] = None,
    title: str = "CFP Playoff Bracket"
):
    """
    Display the 12-team playoff bracket.
    
    Args:
        playoff_teams: DataFrame with seeded teams
        matchups_df: Optional matchups DataFrame
        title: Title for display
    """
    st.subheader(title)
    
    if playoff_teams.empty:
        st.warning("No playoff teams available.")
        return
    
    # Ensure sorted by seed
    playoff_teams = playoff_teams.sort_values("seed").reset_index(drop=True)
    
    # Display teams by seed
    st.write("**Playoff Teams:**")
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Seeds 1-6 (Top Half):**")
        for _, team in playoff_teams.head(6).iterrows():
            seed = team["seed"]
            team_name = team["team"]
            auto_bid = "🏆" if team.get("is_auto_bid", False) else ""
            st.write(f"{seed}. {team_name} {auto_bid}")
    
    with col2:
        st.write("**Seeds 7-12 (Bottom Half):**")
        for _, team in playoff_teams.tail(6).iterrows():
            seed = team["seed"]
            team_name = team["team"]
            auto_bid = "🏆" if team.get("is_auto_bid", False) else ""
            st.write(f"{seed}. {team_name} {auto_bid}")
    
    # Show byes
    byes = playoff_teams[playoff_teams["seed"] <= 4]
    if not byes.empty:
        st.write("**First Round Byes (Seeds 1-4):**")
        for _, team in byes.iterrows():
            st.write(f"  • {team['team']} (Seed {team['seed']})")
    
    # Display matchups if provided
    if matchups_df is not None and not matchups_df.empty:
        st.write("**First Round Matchups:**")
        
        first_round = matchups_df[matchups_df["round"] == "First Round"]
        games = first_round[first_round["team2"].notna()]  # Exclude byes
        
        for _, matchup in games.iterrows():
            team1 = matchup["team1"]
            team1_seed = matchup["team1_seed"]
            team2 = matchup["team2"]
            team2_seed = matchup["team2_seed"]
            
            st.write(f"  • **{team1_seed} {team1}** vs **{team2_seed} {team2}**")
    
    # Show auto-bid info
    auto_bids = playoff_teams[playoff_teams.get("is_auto_bid", False)]
    if not auto_bids.empty:
        st.caption(f"🏆 = Conference Champion (Auto-bid). {len(auto_bids)} auto-bids in field.")


def display_bracket_visual(
    playoff_teams: pd.DataFrame,
    matchups_df: Optional[pd.DataFrame] = None
):
    """
    Display a visual bracket diagram using Plotly.
    
    Args:
        playoff_teams: DataFrame with seeded teams
        matchups_df: Optional matchups DataFrame
    """
    # This is a simplified visual - could be enhanced with full bracket tree
    if playoff_teams.empty:
        return
    
    # Create a simple visualization
    fig = go.Figure()
    
    # Plot teams by seed
    seeds = playoff_teams["seed"].values
    teams = playoff_teams["team"].values
    
    fig.add_trace(go.Scatter(
        x=seeds,
        y=[1] * len(seeds),
        mode='markers+text',
        text=teams,
        textposition="top center",
        marker=dict(size=10, color='blue'),
        name="Playoff Teams"
    ))
    
    fig.update_layout(
        title="CFP Playoff Seeds",
        xaxis_title="Seed",
        yaxis_title="",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_playoff_summary(
    playoff_teams: pd.DataFrame,
    rankings_df: pd.DataFrame
):
    """
    Display summary statistics about the playoff field.
    
    Args:
        playoff_teams: DataFrame with playoff teams
        rankings_df: Full rankings DataFrame
    """
    st.subheader("Playoff Summary")
    
    # Count auto-bids
    auto_bids = playoff_teams[playoff_teams.get("is_auto_bid", False)]
    at_large = playoff_teams[~playoff_teams.get("is_auto_bid", True)]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Teams", len(playoff_teams))
    
    with col2:
        st.metric("Auto-Bids", len(auto_bids))
    
    with col3:
        st.metric("At-Large", len(at_large))
    
    # Show any teams that got in via auto-bid despite low rank
    if not auto_bids.empty:
        low_rank_auto = auto_bids[auto_bids["predicted_rank"] > 12]
        if not low_rank_auto.empty:
            st.warning(
                f"⚠️ {len(low_rank_auto)} conference champion(s) ranked outside Top 12 "
                f"received auto-bids: {', '.join(low_rank_auto['team'].values)}"
            )

