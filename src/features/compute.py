"""
Main feature computation module.

This module orchestrates the computation of all features for teams
at a given point in the season, combining all feature utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

from .utils import (
    compute_record_features,
    compute_sos,
    compute_quality_wins,
    compute_conference_features,
    compute_momentum_features,
    compute_elo_rating,
    compute_statistical_features,
    POWER_5_CONFERENCES
)


def compute_features(
    season: int,
    week: int,
    games_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    rankings_df: Optional[pd.DataFrame] = None,
    champions_df: Optional[pd.DataFrame] = None,
    previous_rankings_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute all features for all teams at a given season/week.
    
    This is the main feature engineering function that combines all
    individual feature computations into a single feature vector per team.
    
    Args:
        season: Season year
        week: Week number (cutoff for games)
        games_df: Processed games DataFrame
        teams_df: Processed teams DataFrame with conference info
        rankings_df: Optional current week rankings (for target variable)
        champions_df: Optional conference champions DataFrame
        previous_rankings_df: Optional previous week's rankings (for Top 25 wins)
        
    Returns:
        DataFrame with one row per team, columns = feature vector + target_rank
    """
    # Filter games up to this week
    week_games = games_df[
        (games_df["season"] == season) &
        (games_df["week"] <= week)
    ].copy()
    
    if week_games.empty:
        return pd.DataFrame()
    
    # Get all unique teams that have played games
    all_teams = set(week_games["team"].unique()) | set(week_games["opponent"].unique())
    
    # Pre-compute team records for efficiency (used by SOS and quality wins)
    team_records = _compute_all_team_records(week_games, season, week)
    
    # Determine if this is final week (championship week)
    # Typically week 15 or 16 for conference championships
    is_final_week = week >= 15
    
    # Compute features for each team
    feature_rows = []
    
    for team in all_teams:
        try:
            features = _compute_team_features(
                team=team,
                season=season,
                week=week,
                games_df=week_games,
                teams_df=teams_df,
                team_records=team_records,
                champions_df=champions_df,
                previous_rankings_df=previous_rankings_df,
                is_final_week=is_final_week
            )
            
            # Add target rank if available
            if rankings_df is not None:
                team_ranking = rankings_df[
                    (rankings_df["season"] == season) &
                    (rankings_df["week"] == week) &
                    (rankings_df["team_id"] == team)
                ]
                if not team_ranking.empty:
                    features["target_rank"] = int(team_ranking.iloc[0]["rank"])
                else:
                    # Unranked teams get rank 26 (or higher)
                    features["target_rank"] = 26
            else:
                features["target_rank"] = None
            
            features["team"] = team
            features["season"] = season
            features["week"] = week
            
            feature_rows.append(features)
            
        except Exception as e:
            # Log error but continue with other teams
            print(f"Warning: Failed to compute features for {team} in {season} week {week}: {e}")
            continue
    
    if not feature_rows:
        return pd.DataFrame()
    
    # Convert to DataFrame
    features_df = pd.DataFrame(feature_rows)
    
    # Compute SOS rank (relative ranking of SOS scores)
    if "sos_score" in features_df.columns:
        features_df["sos_rank"] = features_df["sos_score"].rank(ascending=False, method="min").astype(int)
    
    return features_df


def _compute_team_features(
    team: str,
    season: int,
    week: int,
    games_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    team_records: pd.DataFrame,
    champions_df: Optional[pd.DataFrame],
    previous_rankings_df: Optional[pd.DataFrame],
    is_final_week: bool
) -> Dict:
    """
    Compute all features for a single team.
    
    This is a helper function that orchestrates all feature computations.
    """
    features = {}
    
    # 1. Record features
    record_features = compute_record_features(games_df, team, season, week)
    features.update(record_features)
    
    # 2. Strength of Schedule
    sos_features = compute_sos(games_df, team, season, week, team_records)
    features.update(sos_features)
    
    # 3. Quality wins and bad losses
    quality_features = compute_quality_wins(
        games_df, team, season, week, team_records, previous_rankings_df
    )
    features.update(quality_features)
    
    # 4. Conference features
    conf_features = compute_conference_features(
        teams_df, team, season, champions_df, is_final_week
    )
    features.update(conf_features)
    
    # 5. Momentum features
    momentum_features = compute_momentum_features(games_df, team, season, week)
    features.update(momentum_features)
    
    # 6. Elo rating
    elo = compute_elo_rating(games_df, team, season, week)
    features["elo_rating"] = elo
    
    # 7. Statistical features
    stat_features = compute_statistical_features(games_df, team, season, week)
    features.update(stat_features)
    
    return features


def _compute_all_team_records(
    games_df: pd.DataFrame,
    season: int,
    week: int
) -> pd.DataFrame:
    """
    Pre-compute records for all teams up to each week.
    
    This is more efficient than computing records on-demand for each team.
    
    Returns:
        DataFrame with columns: team, season, as_of_week, wins, losses, games_played, win_pct
    """
    records = []
    
    # Get all unique teams and weeks
    all_teams = set(games_df["team"].unique()) | set(games_df["opponent"].unique())
    all_weeks = sorted(games_df["week"].unique())
    
    for team in all_teams:
        for w in all_weeks:
            if w > week:
                continue
            
            # Get team's games up to this week
            team_games = games_df[
                (games_df["team"] == team) &
                (games_df["week"] <= w) &
                (games_df["team_won"].notna())
            ]
            
            wins = team_games["team_won"].sum()
            games_played = len(team_games)
            losses = games_played - wins
            win_pct = wins / games_played if games_played > 0 else 0.0
            
            records.append({
                "team": team,
                "season": season,
                "as_of_week": w,
                "wins": int(wins),
                "losses": int(losses),
                "games_played": int(games_played),
                "win_pct": win_pct
            })
    
    return pd.DataFrame(records)


def compute_features_for_all_weeks(
    seasons: List[int],
    games_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    champions_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute features for all teams across all weeks in specified seasons.
    
    This generates the full training dataset by computing features for
    every (season, week) combination where rankings exist.
    
    Args:
        seasons: List of seasons to process
        games_df: Processed games DataFrame
        teams_df: Processed teams DataFrame
        rankings_df: Processed rankings DataFrame
        champions_df: Optional conference champions DataFrame
        
    Returns:
        DataFrame with features for all team-week combinations
    """
    all_features = []
    
    # Get all unique (season, week) combinations from rankings
    ranking_weeks = rankings_df[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    
    print(f"  - Computing features for {len(ranking_weeks)} week-season combinations...")
    
    from tqdm import tqdm
    for _, row in tqdm(ranking_weeks.iterrows(), total=len(ranking_weeks), desc="Computing features"):
        season = int(row["season"])
        week = int(row["week"])
        
        if season not in seasons:
            continue
        
        # Get previous week's rankings for Top 25 wins feature
        prev_week = week - 1
        previous_rankings = None
        if prev_week > 0:
            previous_rankings = rankings_df[
                (rankings_df["season"] == season) &
                (rankings_df["week"] == prev_week)
            ]
        
        # Compute features for this week
        week_features = compute_features(
            season=season,
            week=week,
            games_df=games_df,
            teams_df=teams_df,
            rankings_df=rankings_df,
            champions_df=champions_df,
            previous_rankings_df=previous_rankings
        )
        
        if not week_features.empty:
            all_features.append(week_features)
    
    if not all_features:
        return pd.DataFrame()
    
    # Combine all weeks
    full_dataset = pd.concat(all_features, ignore_index=True)
    
    return full_dataset

