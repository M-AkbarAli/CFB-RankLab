"""
Playoff selection and seeding logic.

This module implements the 12-team CFP format rules for selecting
and seeding playoff teams.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def select_playoff_teams(
    rankings_df: pd.DataFrame,
    conference_champions: Dict[str, str],
    format: str = "12team"
) -> pd.DataFrame:
    """
    Select playoff teams based on rankings and conference champions.
    
    Implements 12-team format rules:
    - Top 5 conference champions get auto-bids
    - Remaining 7 spots go to highest-ranked teams
    - Seeds 1-4: Top 4 conference champions
    - Seeds 5-12: Remaining teams by rank
    
    Args:
        rankings_df: DataFrame with predicted rankings (team, predicted_rank)
        conference_champions: Dict mapping conference to champion team
        format: Playoff format ("12team" or "4team")
        
    Returns:
        DataFrame with playoff teams and seeds
    """
    if format == "4team":
        # Old format: just top 4
        playoff_teams = rankings_df.head(4).copy()
        playoff_teams["seed"] = range(1, 5)
        playoff_teams["is_auto_bid"] = False
        return playoff_teams
    
    # 12-team format
    rankings = rankings_df.copy()
    rankings = rankings.sort_values("predicted_rank").reset_index(drop=True)
    
    # Identify conference champions in rankings
    champ_teams = set(conference_champions.values())
    
    # Find top 5 conference champions by rank
    champ_rankings = rankings[rankings["team"].isin(champ_teams)].copy()
    champ_rankings = champ_rankings.sort_values("predicted_rank").head(5)
    
    top_5_champs = set(champ_rankings["team"].values)
    
    # Get remaining teams (non-champs or champs beyond top 5)
    remaining_teams = rankings[~rankings["team"].isin(top_5_champs)].copy()
    
    # Select top 7 remaining teams
    at_large = remaining_teams.head(7)
    
    # Combine: 5 champs + 7 at-large = 12 teams
    playoff_teams = pd.concat([
        champ_rankings,
        at_large
    ], ignore_index=True)
    
    # Mark auto-bids
    playoff_teams["is_auto_bid"] = playoff_teams["team"].isin(top_5_champs)
    
    # Assign seeds
    playoff_teams = assign_seeds(playoff_teams, conference_champions)
    
    return playoff_teams


def assign_seeds(
    playoff_teams: pd.DataFrame,
    conference_champions: Dict[str, str]
) -> pd.DataFrame:
    """
    Assign seeds 1-12 to playoff teams.
    
    Rules:
    - Seeds 1-4: Top 4 conference champions (by rank)
    - Seeds 5-12: Remaining teams sorted by rank
    - Special case: If 5th champ is ranked >12, they get seed #12
    
    Args:
        playoff_teams: DataFrame with playoff teams
        conference_champions: Dict mapping conference to champion
        
    Returns:
        DataFrame with seed column added
    """
    teams = playoff_teams.copy()
    
    # Separate champs and non-champs
    champs = teams[teams["is_auto_bid"]].copy()
    non_champs = teams[~teams["is_auto_bid"]].copy()
    
    # Sort champs by rank
    champs = champs.sort_values("predicted_rank")
    
    # Top 4 champs get seeds 1-4
    top_4_champs = champs.head(4).copy()
    top_4_champs["seed"] = range(1, 5)
    
    # Remaining teams (5th champ + all non-champs) get seeds 5-12
    remaining = pd.concat([
        champs.iloc[4:],  # 5th champ if exists
        non_champs
    ], ignore_index=True)
    remaining = remaining.sort_values("predicted_rank")
    
    # Special case: if 5th champ exists and was ranked >12, they get seed 12
    if len(champs) > 4:
        fifth_champ = champs.iloc[4]
        if fifth_champ["predicted_rank"] > 12:
            # 5th champ gets seed 12, others get 5-11
            remaining_others = remaining[remaining["team"] != fifth_champ["team"]].copy()
            remaining_others["seed"] = range(5, 5 + len(remaining_others))
            fifth_champ_df = pd.DataFrame([fifth_champ])
            fifth_champ_df["seed"] = 12
            remaining = pd.concat([remaining_others, fifth_champ_df], ignore_index=True)
        else:
            remaining["seed"] = range(5, 5 + len(remaining))
    else:
        remaining["seed"] = range(5, 5 + len(remaining))
    
    # Combine
    all_seeded = pd.concat([top_4_champs, remaining], ignore_index=True)
    all_seeded = all_seeded.sort_values("seed")
    
    return all_seeded


def generate_bracket(playoff_teams: pd.DataFrame) -> pd.DataFrame:
    """
    Generate first-round matchups for 12-team bracket.
    
    Format:
    - Seeds 1-4: Byes
    - First round: 12@5, 11@6, 10@7, 9@8
    - Winners advance to play seeds 1-4
    
    Args:
        playoff_teams: DataFrame with seeded teams
        
    Returns:
        DataFrame with matchups
    """
    teams = playoff_teams.copy().sort_values("seed")
    
    # First round matchups
    matchups = []
    
    # Seeds 1-4 get byes
    byes = teams[teams["seed"] <= 4].copy()
    for _, team in byes.iterrows():
        matchups.append({
            "round": "First Round",
            "game": "Bye",
            "team1": team["team"],
            "team1_seed": team["seed"],
            "team2": None,
            "team2_seed": None
        })
    
    # First round games: 12@5, 11@6, 10@7, 9@8
    first_round_pairs = [(12, 5), (11, 6), (10, 7), (9, 8)]
    
    for lower_seed, higher_seed in first_round_pairs:
        lower_team = teams[teams["seed"] == lower_seed]
        higher_team = teams[teams["seed"] == higher_seed]
        
        if not lower_team.empty and not higher_team.empty:
            matchups.append({
                "round": "First Round",
                "game": f"Game {len(matchups) - 4 + 1}",
                "team1": higher_team.iloc[0]["team"],
                "team1_seed": higher_seed,
                "team2": lower_team.iloc[0]["team"],
                "team2_seed": lower_seed
            })
    
    return pd.DataFrame(matchups)


def determine_conference_champions(
    games_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    season: int,
    week: int,
    user_champions: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Determine conference champions from game results.
    
    This is a simplified version that:
    1. Uses user-specified champions if provided
    2. Otherwise, determines from conference championship game results
    3. Falls back to division winners if no CCG
    
    Args:
        games_df: Games DataFrame
        teams_df: Teams DataFrame
        season: Season year
        week: Week number (should be championship week)
        user_champions: Optional dict of {conference: champion} from user
        
    Returns:
        Dict mapping conference to champion team
    """
    if user_champions:
        return user_champions
    
    champions = {}
    
    # Get conference championship games (typically week 15)
    ccg_games = games_df[
        (games_df["season"] == season) &
        (games_df["week"] == week) &
        (games_df.get("is_conference_championship", False) | 
         games_df.get("conference_championship", False))
    ]
    
    # Determine champions from CCG results
    for _, game in ccg_games.iterrows():
        # Get conference from teams
        winner = game["team"] if game["team_won"] else game["opponent"]
        
        # Find winner's conference
        winner_info = teams_df[teams_df["team_id"] == winner]
        if not winner_info.empty:
            conf = winner_info.iloc[0].get("conference")
            if conf:
                champions[conf] = winner
    
    # For conferences without CCG or if week < 15, use division winners
    # This is simplified - in practice would need division standings
    # For now, return what we have
    
    return champions

