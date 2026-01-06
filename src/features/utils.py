"""
Feature computation utility functions.

This module contains helper functions for computing individual features
that mirror the CFP committee's evaluation criteria.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set


# Power 5 conferences (historical, pre-2024 realignment)
POWER_5_CONFERENCES = {
    "SEC", "Big Ten", "Big 12", "ACC", "Pac-12"
}


def compute_record_features(
    games_df: pd.DataFrame,
    team: str,
    season: int,
    week: int
) -> Dict[str, float]:
    """
    Compute basic win-loss record features.
    
    Args:
        games_df: Processed games DataFrame
        team: Team name
        season: Season year
        week: Week number (cutoff)
        
    Returns:
        Dictionary with record features
    """
    # Filter games for this team up to this week
    team_games = games_df[
        (games_df["team"] == team) &
        (games_df["season"] == season) &
        (games_df["week"] <= week) &
        (games_df["team_won"].notna())
    ]
    
    wins = team_games["team_won"].sum()
    losses = len(team_games) - wins
    games_played = len(team_games)
    win_pct = wins / games_played if games_played > 0 else 0.0
    
    # Conference vs non-conference
    conf_games = team_games[team_games.get("is_conference_game", False)]
    conf_wins = conf_games["team_won"].sum() if not conf_games.empty else 0
    conf_losses = len(conf_games) - conf_wins
    
    non_conf_games = team_games[~team_games.get("is_conference_game", True)]
    non_conf_wins = non_conf_games["team_won"].sum() if not non_conf_games.empty else 0
    non_conf_losses = len(non_conf_games) - non_conf_wins
    
    return {
        "wins": float(wins),
        "losses": float(losses),
        "games_played": float(games_played),
        "win_pct": win_pct,
        "conference_wins": float(conf_wins),
        "conference_losses": float(conf_losses),
        "non_conference_wins": float(non_conf_wins),
        "non_conference_losses": float(non_conf_losses)
    }


def compute_sos(
    games_df: pd.DataFrame,
    team: str,
    season: int,
    week: int,
    team_records: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Compute Strength of Schedule metrics.
    
    Args:
        games_df: Processed games DataFrame
        team: Team name
        season: Season year
        week: Week number (cutoff)
        team_records: Optional pre-computed team records DataFrame
        
    Returns:
        Dictionary with SOS features
    """
    # Get team's games
    team_games = games_df[
        (games_df["team"] == team) &
        (games_df["season"] == season) &
        (games_df["week"] <= week)
    ]
    
    if team_games.empty:
        return {
            "sos_score": 0.0,
            "opponents_avg_wins": 0.0,
            "opponents_avg_win_pct": 0.0
        }
    
    # Compute opponent records
    if team_records is None:
        # Compute records on the fly (less efficient)
        opponent_records = []
        for _, game in team_games.iterrows():
            opp = game["opponent"]
            opp_games = games_df[
                (games_df["team"] == opp) &
                (games_df["season"] == season) &
                (games_df["week"] <= game["week"]) &
                (games_df["team_won"].notna())
            ]
            if not opp_games.empty:
                opp_wins = opp_games["team_won"].sum()
                opp_games_played = len(opp_games)
                opponent_records.append({
                    "opponent": opp,
                    "wins": opp_wins,
                    "games_played": opp_games_played,
                    "win_pct": opp_wins / opp_games_played if opp_games_played > 0 else 0.0
                })
    else:
        # Use pre-computed records
        opponent_records = []
        for _, game in team_games.iterrows():
            opp = game["opponent"]
            opp_week = game["week"]
            opp_record = team_records[
                (team_records["team"] == opp) &
                (team_records["season"] == season) &
                (team_records["as_of_week"] == opp_week)
            ]
            if not opp_record.empty:
                rec = opp_record.iloc[0]
                opponent_records.append({
                    "opponent": opp,
                    "wins": rec["wins"],
                    "games_played": rec["games_played"],
                    "win_pct": rec["win_pct"]
                })
    
    if not opponent_records:
        return {
            "sos_score": 0.0,
            "opponents_avg_wins": 0.0,
            "opponents_avg_win_pct": 0.0
        }
    
    opp_df = pd.DataFrame(opponent_records)
    
    return {
        "sos_score": float(opp_df["win_pct"].mean()),
        "opponents_avg_wins": float(opp_df["wins"].mean()),
        "opponents_avg_win_pct": float(opp_df["win_pct"].mean())
    }


def compute_quality_wins(
    games_df: pd.DataFrame,
    team: str,
    season: int,
    week: int,
    team_records: Optional[pd.DataFrame] = None,
    previous_rankings: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Compute quality wins and bad losses metrics.
    
    Args:
        games_df: Processed games DataFrame
        team: Team name
        season: Season year
        week: Week number (cutoff)
        team_records: Optional pre-computed team records
        previous_rankings: Optional previous week's rankings for Top 25 wins
        
    Returns:
        Dictionary with quality win/bad loss features
    """
    # Get team's wins and losses
    team_games = games_df[
        (games_df["team"] == team) &
        (games_df["season"] == season) &
        (games_df["week"] <= week) &
        (games_df["team_won"].notna())
    ]
    
    wins = team_games[team_games["team_won"] == True]
    losses = team_games[team_games["team_won"] == False]
    
    # Quality wins: wins vs teams with winning records
    wins_vs_winning = 0
    wins_vs_power5 = 0
    wins_vs_top25 = 0
    notable_wins = 0  # Wins vs teams with 8+ wins
    
    for _, game in wins.iterrows():
        opp = game["opponent"]
        
        # Check if opponent is Power 5
        if game.get("opp_conference") in POWER_5_CONFERENCES:
            wins_vs_power5 += 1
        
        # Check opponent's record
        if team_records is not None:
            opp_record = team_records[
                (team_records["team"] == opp) &
                (team_records["season"] == season) &
                (team_records["as_of_week"] <= game["week"])
            ]
            if not opp_record.empty:
                opp_wins = opp_record.iloc[-1]["wins"]
                opp_games = opp_record.iloc[-1]["games_played"]
                opp_win_pct = opp_wins / opp_games if opp_games > 0 else 0.0
                
                if opp_win_pct > 0.5:
                    wins_vs_winning += 1
                if opp_wins >= 8:
                    notable_wins += 1
        
        # Check if opponent was in previous week's Top 25
        if previous_rankings is not None:
            prev_week = week - 1 if week > 1 else week
            opp_ranked = previous_rankings[
                (previous_rankings["team_id"] == opp) &
                (previous_rankings["season"] == season) &
                (previous_rankings["week"] == prev_week)
            ]
            if not opp_ranked.empty:
                wins_vs_top25 += 1
    
    # Bad losses: losses to sub-.500 teams
    bad_losses = 0
    losses_vs_top10 = 0
    
    for _, game in losses.iterrows():
        opp = game["opponent"]
        
        # Check opponent's record
        if team_records is not None:
            opp_record = team_records[
                (team_records["team"] == opp) &
                (team_records["season"] == season) &
                (team_records["as_of_week"] <= game["week"])
            ]
            if not opp_record.empty:
                opp_wins = opp_record.iloc[-1]["wins"]
                opp_games = opp_record.iloc[-1]["games_played"]
                opp_win_pct = opp_wins / opp_games if opp_games > 0 else 0.0
                
                if opp_win_pct < 0.5:
                    bad_losses += 1
        
        # Check if opponent was in Top 10 (less penalized)
        if previous_rankings is not None:
            prev_week = week - 1 if week > 1 else week
            opp_ranked = previous_rankings[
                (previous_rankings["team_id"] == opp) &
                (previous_rankings["season"] == season) &
                (previous_rankings["week"] == prev_week) &
                (previous_rankings["rank"] <= 10)
            ]
            if not opp_ranked.empty:
                losses_vs_top10 += 1
    
    return {
        "wins_vs_winning_teams": float(wins_vs_winning),
        "wins_vs_power5": float(wins_vs_power5),
        "wins_vs_top25": float(wins_vs_top25),
        "notable_wins": float(notable_wins),
        "bad_losses": float(bad_losses),
        "losses_vs_top10": float(losses_vs_top10)
    }


def compute_conference_features(
    teams_df: pd.DataFrame,
    team: str,
    season: int,
    champions_df: Optional[pd.DataFrame] = None,
    is_final_week: bool = False
) -> Dict[str, any]:
    """
    Compute conference-related features.
    
    Args:
        teams_df: Processed teams DataFrame
        team: Team name
        season: Season year
        champions_df: Optional conference champions DataFrame
        is_final_week: Whether this is the final ranking (championship week)
        
    Returns:
        Dictionary with conference features
    """
    # Get team's conference
    team_info = teams_df[
        (teams_df["team_id"] == team) &
        (teams_df["season"] == season)
    ]
    
    if team_info.empty:
        # Try without season filter
        team_info = teams_df[teams_df["team_id"] == team]
    
    conference = team_info["conference"].iloc[0] if not team_info.empty and "conference" in team_info.columns else None
    is_power5 = conference in POWER_5_CONFERENCES if conference else False
    
    # Check if conference champion
    is_champion = False
    if champions_df is not None and is_final_week:
        champ_check = champions_df[
            (champions_df["season"] == season) &
            (champions_df["champion_team_id"] == team)
        ]
        is_champion = not champ_check.empty
    
    return {
        "conference": conference if conference else "Unknown",
        "is_power5": bool(is_power5),
        "is_conference_champion": bool(is_champion)
    }


def compute_momentum_features(
    games_df: pd.DataFrame,
    team: str,
    season: int,
    week: int
) -> Dict[str, float]:
    """
    Compute momentum and recency features.
    
    Args:
        games_df: Processed games DataFrame
        team: Team name
        season: Season year
        week: Week number (cutoff)
        
    Returns:
        Dictionary with momentum features
    """
    # Get team's games, sorted by week
    team_games = games_df[
        (games_df["team"] == team) &
        (games_df["season"] == season) &
        (games_df["week"] <= week) &
        (games_df["team_won"].notna())
    ].sort_values("week")
    
    if team_games.empty:
        return {
            "current_win_streak": 0.0,
            "last_game_result": 0.0,  # 0 = loss, 1 = win
            "last_game_opponent_quality": 0.0
        }
    
    # Current win streak (count backwards from most recent)
    win_streak = 0
    for _, game in team_games.iloc[::-1].iterrows():
        if game["team_won"]:
            win_streak += 1
        else:
            break
    
    # Last game result
    last_game = team_games.iloc[-1]
    last_result = 1.0 if last_game["team_won"] else 0.0
    
    # Last game opponent quality (simplified: Power 5 = 1, else 0.5)
    last_opp_quality = 0.5
    if last_game.get("opp_conference") in POWER_5_CONFERENCES:
        last_opp_quality = 1.0
    
    return {
        "current_win_streak": float(win_streak),
        "last_game_result": last_result,
        "last_game_opponent_quality": last_opp_quality
    }


def compute_elo_rating(
    games_df: pd.DataFrame,
    team: str,
    season: int,
    week: int,
    initial_elo: float = 1500.0,
    k_factor: float = 32.0
) -> float:
    """
    Compute Elo rating for a team as of a specific week.
    
    Args:
        games_df: Processed games DataFrame
        team: Team name
        season: Season year
        week: Week number (cutoff)
        initial_elo: Starting Elo rating
        k_factor: K-factor for Elo updates
        
    Returns:
        Elo rating as float
    """
    # Get team's games in order
    team_games = games_df[
        (games_df["team"] == team) &
        (games_df["season"] == season) &
        (games_df["week"] <= week) &
        (games_df["team_won"].notna())
    ].sort_values("week")
    
    if team_games.empty:
        return initial_elo
    
    # Simple Elo calculation (would need opponent Elo for full accuracy)
    # For now, use a simplified version
    elo = initial_elo
    
    for _, game in team_games.iterrows():
        # Simplified: win = +k, loss = -k (would need opponent Elo for proper calculation)
        if game["team_won"]:
            elo += k_factor * 0.5  # Simplified update
        else:
            elo -= k_factor * 0.5
    
    return float(elo)


def compute_statistical_features(
    games_df: pd.DataFrame,
    team: str,
    season: int,
    week: int
) -> Dict[str, float]:
    """
    Compute basic statistical features (points, margins).
    
    Note: Committee explicitly avoids using margin of victory as primary factor,
    but we include it as a subtle feature.
    
    Args:
        games_df: Processed games DataFrame
        team: Team name
        season: Season year
        week: Week number (cutoff)
        
    Returns:
        Dictionary with statistical features
    """
    # Get team's games
    team_games = games_df[
        (games_df["team"] == team) &
        (games_df["season"] == season) &
        (games_df["week"] <= week) &
        (games_df["team_score"].notna()) &
        (games_df["opp_score"].notna())
    ]
    
    if team_games.empty:
        return {
            "points_per_game": 0.0,
            "points_allowed_per_game": 0.0,
            "point_differential": 0.0
        }
    
    points_scored = team_games["team_score"].sum()
    points_allowed = team_games["opp_score"].sum()
    games_played = len(team_games)
    
    return {
        "points_per_game": float(points_scored / games_played),
        "points_allowed_per_game": float(points_allowed / games_played),
        "point_differential": float((points_scored - points_allowed) / games_played)
    }

