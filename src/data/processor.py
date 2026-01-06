"""
Data processing module for cleaning and structuring raw API data.

This module transforms raw API responses into structured DataFrames
suitable for feature engineering and modeling.
"""

import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import pickle


class DataProcessor:
    """Processes raw API data into structured formats."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def process_rankings(
        self, 
        rankings_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process and standardize rankings data.
        
        Args:
            rankings_df: Raw rankings DataFrame from fetcher
            
        Returns:
            Processed DataFrame with columns: season, week, team_id, rank
        """
        if rankings_df.empty:
            return pd.DataFrame(columns=["season", "week", "team_id", "rank"])
        
        df = rankings_df.copy()
        
        # Ensure required columns exist
        required_cols = ["season", "week", "team_id", "rank"]
        for col in required_cols:
            if col not in df.columns:
                if col == "team_id" and "team" in df.columns:
                    df["team_id"] = df["team"]
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # Clean data
        df = df[df["rank"].notna()].copy()
        df["rank"] = df["rank"].astype(int)
        df["week"] = df["week"].astype(int)
        df["season"] = df["season"].astype(int)
        
        # Sort by season, week, rank
        df = df.sort_values(["season", "week", "rank"]).reset_index(drop=True)
        
        return df[required_cols]
    
    def process_games(
        self, 
        games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process and standardize game results data.
        
        Args:
            games_df: Raw games DataFrame from fetcher
            
        Returns:
            Processed DataFrame with standardized game data
        """
        if games_df.empty:
            return pd.DataFrame()
        
        df = games_df.copy()
        
        # Ensure date is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
        
        # Create a unified games format with team/opponent structure
        # This creates two rows per game (one for each team's perspective)
        team_games = []
        
        for _, game in df.iterrows():
            game_id = game.get("game_id", f"{game.get('season')}_{game.get('week')}_{game.get('home_team')}")
            season = game.get("season")
            week = game.get("week")
            date = game.get("date")
            
            # Home team perspective
            if pd.notna(game.get("home_team")):
                team_games.append({
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "date": date,
                    "team": game["home_team"],
                    "opponent": game.get("away_team"),
                    "team_score": game.get("home_score"),
                    "opp_score": game.get("away_score"),
                    "location": "home" if not game.get("neutral_site", False) else "neutral",
                    "conference": game.get("home_conference"),
                    "opp_conference": game.get("away_conference"),
                    "is_conference_game": game.get("home_conference") == game.get("away_conference"),
                    "winner": game.get("winner"),
                    "team_won": game.get("winner") == game.get("home_team") if pd.notna(game.get("winner")) else None
                })
            
            # Away team perspective
            if pd.notna(game.get("away_team")):
                team_games.append({
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "date": date,
                    "team": game["away_team"],
                    "opponent": game.get("home_team"),
                    "team_score": game.get("away_score"),
                    "opp_score": game.get("home_score"),
                    "location": "away" if not game.get("neutral_site", False) else "neutral",
                    "conference": game.get("away_conference"),
                    "opp_conference": game.get("home_conference"),
                    "is_conference_game": game.get("home_conference") == game.get("away_conference"),
                    "winner": game.get("winner"),
                    "team_won": game.get("winner") == game.get("away_team") if pd.notna(game.get("winner")) else None
                })
        
        result_df = pd.DataFrame(team_games)
        
        # Clean and type conversions
        if "team_won" in result_df.columns:
            result_df["team_won"] = result_df["team_won"].astype(bool)
        
        # Sort by date, then week
        if "date" in result_df.columns:
            result_df = result_df.sort_values(["season", "date", "week"]).reset_index(drop=True)
        else:
            result_df = result_df.sort_values(["season", "week"]).reset_index(drop=True)
        
        return result_df
    
    def process_teams(
        self, 
        teams_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process and standardize team metadata.
        
        Args:
            teams_df: Raw teams DataFrame from fetcher
            
        Returns:
            Processed DataFrame with team information
        """
        if teams_df.empty:
            return pd.DataFrame()
        
        df = teams_df.copy()
        
        # Standardize column names
        if "team_id" not in df.columns and "team_name" in df.columns:
            df["team_id"] = df["team_name"]
        
        # Ensure we have team_id
        if "team_id" not in df.columns:
            raise ValueError("Teams DataFrame must have team_id or team_name column")
        
        # Select relevant columns
        cols = ["team_id"]
        if "team_name" in df.columns:
            cols.append("team_name")
        if "conference" in df.columns:
            cols.append("conference")
        if "season" in df.columns:
            cols.append("season")
        
        result = df[cols].copy()
        
        # Remove duplicates (keep first occurrence)
        result = result.drop_duplicates(subset=["team_id", "season"] if "season" in result.columns else ["team_id"])
        
        return result
    
    def process_champions(
        self, 
        champions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process conference champions data.
        
        Args:
            champions_df: Raw champions DataFrame from fetcher
            
        Returns:
            Processed DataFrame with conference champions
        """
        if champions_df.empty:
            return pd.DataFrame(columns=["season", "conference", "champion_team_id"])
        
        df = champions_df.copy()
        
        # Standardize column names
        if "champion_team" in df.columns:
            df["champion_team_id"] = df["champion_team"]
        
        required_cols = ["season", "conference", "champion_team_id"]
        available_cols = [col for col in required_cols if col in df.columns]
        
        return df[available_cols].copy()
    
    def map_ranking_weeks_to_games(
        self, 
        rankings_df: pd.DataFrame, 
        games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Map ranking release weeks to the games that were included.
        
        This is critical for feature engineering - we need to know
        which games were considered in each ranking.
        
        Args:
            rankings_df: Processed rankings DataFrame
            games_df: Processed games DataFrame
            
        Returns:
            DataFrame mapping (season, week) to cutoff date
        """
        # For each ranking week, find the latest game date
        week_cutoffs = []
        
        for (season, week), group in rankings_df.groupby(["season", "week"]):
            # Find games up to this week
            season_games = games_df[games_df["season"] == season]
            week_games = season_games[season_games["week"] <= week]
            
            if not week_games.empty and "date" in week_games.columns:
                cutoff_date = week_games["date"].max()
            else:
                cutoff_date = None
            
            week_cutoffs.append({
                "season": season,
                "week": week,
                "cutoff_date": cutoff_date,
                "games_included": len(week_games)
            })
        
        return pd.DataFrame(week_cutoffs)
    
    def get_team_records_at_week(
        self, 
        games_df: pd.DataFrame, 
        season: int, 
        week: int
    ) -> pd.DataFrame:
        """
        Calculate team records as of a specific week.
        
        Args:
            games_df: Processed games DataFrame
            season: Season year
            week: Week number
            
        Returns:
            DataFrame with team records (wins, losses, win_pct)
        """
        # Filter games up to the specified week
        relevant_games = games_df[
            (games_df["season"] == season) & 
            (games_df["week"] <= week) &
            (games_df["team_won"].notna())
        ].copy()
        
        # Calculate wins and losses
        records = relevant_games.groupby("team").agg({
            "team_won": ["sum", "count"]
        }).reset_index()
        
        records.columns = ["team", "wins", "games_played"]
        records["losses"] = records["games_played"] - records["wins"]
        records["win_pct"] = records["wins"] / records["games_played"].replace(0, 1)
        records["season"] = season
        records["as_of_week"] = week
        
        return records[["season", "as_of_week", "team", "wins", "losses", "games_played", "win_pct"]]
    
    def save_processed_data(
        self, 
        data_dict: Dict[str, pd.DataFrame], 
        filename: str = "processed_data.pkl"
    ):
        """Save processed data to disk."""
        filepath = self.processed_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Saved processed data to {filepath}")
    
    def load_processed_data(self, filename: str = "processed_data.pkl") -> Dict[str, pd.DataFrame]:
        """Load processed data from disk."""
        filepath = self.processed_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Processed data file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data


if __name__ == "__main__":
    # Example usage
    from src.data.fetcher import CFBDFetcher
    
    fetcher = CFBDFetcher()
    processor = DataProcessor()
    
    # Fetch and process sample data
    print("Fetching sample data...")
    raw_data = fetcher.fetch_all_historical_data(2023, 2023)
    
    print("Processing data...")
    processed_rankings = processor.process_rankings(raw_data["rankings"])
    processed_games = processor.process_games(raw_data["games"])
    processed_teams = processor.process_teams(raw_data["teams"])
    processed_champions = processor.process_champions(raw_data["champions"])
    
    print(f"Processed {len(processed_rankings)} ranking entries")
    print(f"Processed {len(processed_games)} game records")
    print(f"Processed {len(processed_teams)} teams")
    print(f"Processed {len(processed_champions)} conference champions")

