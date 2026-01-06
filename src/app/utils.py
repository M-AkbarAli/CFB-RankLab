"""
Utility functions for Streamlit app.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from ..data.fetcher import CFBDFetcher
from ..data.processor import DataProcessor
from ..features.compute import compute_features
from ..simulation.engine import SimulationEngine
from ..simulation.playoff import select_playoff_teams, generate_bracket, determine_conference_champions


def load_current_season_data(
    season: int,
    cache_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load data for the current season.
    
    Args:
        season: Season year
        cache_dir: Optional cache directory
        
    Returns:
        Dictionary with keys: games, teams, rankings, champions
    """
    fetcher = CFBDFetcher()
    processor = DataProcessor()
    
    # Fetch current season data
    games = fetcher.fetch_games(season, season_type="regular")
    teams = fetcher.fetch_teams(season)
    rankings = fetcher.fetch_rankings(season)
    champions = fetcher.fetch_conference_champions(season)
    
    # Process
    games_df = processor.process_games(games)
    teams_df = processor.process_teams(teams)
    rankings_df = processor.process_rankings(rankings)
    champions_df = processor.process_champions(champions) if not champions.empty else pd.DataFrame()
    
    return {
        "games": games_df,
        "teams": teams_df,
        "rankings": rankings_df,
        "champions": champions_df
    }


def get_current_week(season: int, games_df: pd.DataFrame) -> int:
    """
    Determine the current week based on game dates.
    
    Args:
        season: Season year
        games_df: Games DataFrame
        
    Returns:
        Current week number
    """
    season_games = games_df[games_df["season"] == season]
    
    if season_games.empty:
        return 1
    
    # Get most recent game date
    if "date" in season_games.columns:
        latest_date = season_games["date"].max()
        latest_week = season_games[season_games["date"] == latest_date]["week"].iloc[0]
        return int(latest_week)
    
    # Fallback: use max week
    return int(season_games["week"].max())


def run_simulation(
    games_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    game_outcomes: Dict[str, str],
    season: int,
    target_week: int,
    model_path: Optional[Path] = None,
    champions_df: Optional[pd.DataFrame] = None
) -> Dict[str, pd.DataFrame]:
    """
    Run a complete simulation and return results.
    
    Args:
        games_df: Base games DataFrame
        teams_df: Teams DataFrame
        game_outcomes: Dict of game_id -> winner
        season: Season year
        target_week: Target week for projection
        model_path: Optional model path
        champions_df: Optional champions DataFrame
        
    Returns:
        Dictionary with keys: rankings, playoff_teams, matchups
    """
    # Initialize simulation engine
    engine = SimulationEngine(model_path=model_path)
    
    # Run simulation
    rankings = engine.simulate_scenario(
        base_games_df=games_df,
        base_teams_df=teams_df,
        game_outcomes=game_outcomes,
        target_week=target_week,
        season=season,
        champions_df=champions_df
    )
    
    if rankings.empty:
        return {
            "rankings": pd.DataFrame(),
            "playoff_teams": pd.DataFrame(),
            "matchups": pd.DataFrame()
        }
    
    # Determine conference champions
    conf_champions = determine_conference_champions(
        games_df=games_df,
        teams_df=teams_df,
        season=season,
        week=target_week,
        user_champions=None  # Could be passed in
    )
    
    # Select playoff teams
    playoff_teams = select_playoff_teams(
        rankings_df=rankings,
        conference_champions=conf_champions,
        format="12team"
    )
    
    # Generate bracket
    matchups = generate_bracket(playoff_teams)
    
    return {
        "rankings": rankings,
        "playoff_teams": playoff_teams,
        "matchups": matchups
    }

