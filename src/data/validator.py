"""
Data validation module to check data quality and consistency.
"""

import pandas as pd
from typing import List, Dict, Tuple


class DataValidator:
    """Validates data quality and consistency."""
    
    @staticmethod
    def validate_rankings(rankings_df: pd.DataFrame) -> List[str]:
        """
        Validate rankings data.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if rankings_df.empty:
            return ["Rankings DataFrame is empty"]
        
        # Check required columns
        required_cols = ["season", "week", "team_id", "rank"]
        for col in required_cols:
            if col not in rankings_df.columns:
                errors.append(f"Missing required column: {col}")
        
        if errors:
            return errors
        
        # Check for duplicate ranks in same week
        duplicates = rankings_df.groupby(["season", "week", "rank"]).size()
        duplicates = duplicates[duplicates > 1]
        if not duplicates.empty:
            errors.append(f"Found duplicate ranks: {duplicates.to_dict()}")
        
        # Check rank values are positive integers
        invalid_ranks = rankings_df[~rankings_df["rank"].between(1, 25)]
        if not invalid_ranks.empty:
            errors.append(f"Found invalid rank values: {invalid_ranks['rank'].unique()}")
        
        # Check for missing values
        missing = rankings_df[required_cols].isnull().sum()
        if missing.any():
            errors.append(f"Missing values found: {missing[missing > 0].to_dict()}")
        
        return errors
    
    @staticmethod
    def validate_games(games_df: pd.DataFrame) -> List[str]:
        """
        Validate games data.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if games_df.empty:
            return ["Games DataFrame is empty"]
        
        # Check required columns
        required_cols = ["season", "team", "opponent", "team_won"]
        for col in required_cols:
            if col not in games_df.columns:
                errors.append(f"Missing required column: {col}")
        
        if errors:
            return errors
        
        # Check for games where both teams won (impossible)
        invalid_wins = games_df.groupby("game_id")["team_won"].sum()
        invalid_wins = invalid_wins[invalid_wins != 1]
        if not invalid_wins.empty:
            errors.append(f"Found games with invalid win counts: {len(invalid_wins)} games")
        
        # Check for missing scores where winner is determined
        if "team_score" in games_df.columns and "opp_score" in games_df.columns:
            score_mismatch = games_df[
                (games_df["team_won"] == True) & 
                (games_df["team_score"] <= games_df["opp_score"])
            ]
            if not score_mismatch.empty:
                errors.append(f"Found {len(score_mismatch)} games with score/winner mismatch")
        
        return errors
    
    @staticmethod
    def validate_team_records(
        games_df: pd.DataFrame, 
        records_df: pd.DataFrame
    ) -> List[str]:
        """
        Validate that computed team records match game results.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # For each team/season/week, verify wins + losses = games played
        for _, record in records_df.iterrows():
            team = record["team"]
            season = record["season"]
            week = record["as_of_week"]
            
            # Get games for this team up to this week
            team_games = games_df[
                (games_df["team"] == team) &
                (games_df["season"] == season) &
                (games_df["week"] <= week) &
                (games_df["team_won"].notna())
            ]
            
            expected_wins = team_games["team_won"].sum()
            expected_losses = len(team_games) - expected_wins
            expected_games = len(team_games)
            
            if record["wins"] != expected_wins:
                errors.append(
                    f"Mismatch for {team} in {season} week {week}: "
                    f"expected {expected_wins} wins, got {record['wins']}"
                )
            
            if record["losses"] != expected_losses:
                errors.append(
                    f"Mismatch for {team} in {season} week {week}: "
                    f"expected {expected_losses} losses, got {record['losses']}"
                )
        
        return errors
    
    @staticmethod
    def validate_data_consistency(
        rankings_df: pd.DataFrame,
        games_df: pd.DataFrame,
        teams_df: pd.DataFrame
    ) -> List[str]:
        """
        Validate consistency across datasets.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check that all teams in rankings exist in teams_df
        ranked_teams = set(rankings_df["team_id"].unique())
        known_teams = set(teams_df["team_id"].unique())
        unknown_teams = ranked_teams - known_teams
        if unknown_teams:
            errors.append(f"Found {len(unknown_teams)} teams in rankings not in teams data: {list(unknown_teams)[:5]}")
        
        # Check that all teams in games exist in teams_df
        game_teams = set(games_df["team"].unique()) | set(games_df["opponent"].unique())
        unknown_game_teams = game_teams - known_teams
        if unknown_game_teams:
            errors.append(f"Found {len(unknown_game_teams)} teams in games not in teams data: {list(unknown_game_teams)[:5]}")
        
        # Check season overlap
        ranking_seasons = set(rankings_df["season"].unique())
        game_seasons = set(games_df["season"].unique())
        if ranking_seasons != game_seasons:
            missing = ranking_seasons - game_seasons
            if missing:
                errors.append(f"Rankings exist for seasons without games: {missing}")
        
        return errors

