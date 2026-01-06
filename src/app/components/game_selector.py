"""
Game selector component for Streamlit app.

Allows users to select winners for remaining games.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Optional


def game_selector(
    games_df: pd.DataFrame,
    current_week: int,
    season: int,
    default_outcomes: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Display game selector UI and return user-selected outcomes.
    
    Args:
        games_df: Games DataFrame
        current_week: Current week number
        season: Season year
        default_outcomes: Optional dict of default winners
        
    Returns:
        Dict mapping game_id to winner team name
    """
    # Filter to remaining games (future weeks)
    remaining_games = games_df[
        (games_df["season"] == season) &
        (games_df["week"] > current_week)
    ].copy()
    
    if remaining_games.empty:
        st.info("No remaining games to select.")
        return {}
    
    # Group by week
    selected_outcomes = {}
    
    st.subheader("Select Game Outcomes")
    
    weeks = sorted(remaining_games["week"].unique())
    
    for week in weeks:
        week_games = remaining_games[remaining_games["week"] == week]
        
        with st.expander(f"Week {week} ({len(week_games)} games)"):
            for idx, game in week_games.iterrows():
                game_id = game.get("game_id", f"{season}_{week}_{game['team']}_{game['opponent']}")
                team1 = game["team"]
                team2 = game["opponent"]
                
                # Determine default winner
                default_winner = None
                if default_outcomes and game_id in default_outcomes:
                    default_winner = default_outcomes[game_id]
                elif game.get("team_won") is not None:
                    # Use actual result if available
                    default_winner = team1 if game["team_won"] else team2
                else:
                    # Default to team1
                    default_winner = team1
                
                # Radio button for winner selection
                winner = st.radio(
                    f"{team1} vs {team2}",
                    options=[team1, team2],
                    index=0 if default_winner == team1 else 1,
                    key=f"game_{game_id}",
                    horizontal=True
                )
                
                selected_outcomes[game_id] = winner
    
    return selected_outcomes


def conference_championship_selector(
    games_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    season: int,
    week: int = 15
) -> Dict[str, str]:
    """
    Special selector for conference championship games.
    
    Args:
        games_df: Games DataFrame
        teams_df: Teams DataFrame
        season: Season year
        week: Championship week (default 15)
        
    Returns:
        Dict mapping conference to champion team
    """
    # Find conference championship games
    ccg_games = games_df[
        (games_df["season"] == season) &
        (games_df["week"] == week)
    ]
    
    # Group by conference (simplified - would need conference info)
    champions = {}
    
    st.subheader("Conference Championships")
    
    for idx, game in ccg_games.iterrows():
        team1 = game["team"]
        team2 = game["opponent"]
        
        # Try to determine conference from teams
        team1_info = teams_df[teams_df["team_id"] == team1]
        team2_info = teams_df[teams_df["team_id"] == team2]
        
        conf = None
        if not team1_info.empty:
            conf = team1_info.iloc[0].get("conference")
        elif not team2_info.empty:
            conf = team2_info.iloc[0].get("conference")
        
        conf_label = conf if conf else f"Conference {idx}"
        
        winner = st.radio(
            f"{conf_label} Championship: {team1} vs {team2}",
            options=[team1, team2],
            key=f"ccg_{idx}",
            horizontal=True
        )
        
        if conf:
            champions[conf] = winner
    
    return champions

