"""
CFBD API client for fetching college football data.

This module handles all interactions with the CollegeFootballData API,
including rankings, games, teams, and conference champions.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Base URL for CFBD API
CFBD_BASE_URL = "https://api.collegefootballdata.com"

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class CFBDFetcher:
    """Client for fetching data from CollegeFootballData API."""
    
    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        """
        Initialize the CFBD API client.
        
        Args:
            api_key: CFBD API key. If None, reads from CFBD_API_KEY env var.
            use_cache: Whether to cache API responses locally.
        """
        self.api_key = api_key or os.getenv("CFBD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "CFBD API key required. Set CFBD_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.use_cache = use_cache
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        self.rate_limit_delay = 0.1  # Delay between requests to respect rate limits
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Make an API request with caching and error handling.
        
        Args:
            endpoint: API endpoint (e.g., '/rankings')
            params: Query parameters
            
        Returns:
            JSON response as list of dictionaries
        """
        url = f"{CFBD_BASE_URL}{endpoint}"
        cache_key = self._get_cache_key(endpoint, params)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        # Check cache first
        if self.use_cache and cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Make API request
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            if self.use_cache:
                with open(cache_file, 'w') as f:
                    json.dump(data, f, indent=2)
            
            return data
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed for {endpoint}: {str(e)}")
    
    def _get_cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate a cache key from endpoint and parameters."""
        key_parts = [endpoint.replace('/', '_').strip('_')]
        if params:
            sorted_params = sorted(params.items())
            param_str = '_'.join(f"{k}_{v}" for k, v in sorted_params)
            key_parts.append(param_str)
        return '_'.join(key_parts)
    
    def fetch_rankings(
        self, 
        year: int, 
        week: Optional[int] = None,
        season_type: str = "regular"
    ) -> pd.DataFrame:
        """
        Fetch CFP rankings for a given year and optional week.
        
        Args:
            year: Season year
            week: Week number (None for all weeks)
            season_type: 'regular' or 'postseason'
            
        Returns:
            DataFrame with columns: season, week, team_id, team, rank
        """
        params = {
            "year": year,
            "seasonType": season_type
        }
        
        data = self._make_request("/rankings", params)
        
        # Filter by week if specified
        if week is not None:
            data = [r for r in data if r.get("week") == week]
        
        # Flatten the rankings data
        records = []
        for ranking_set in data:
            season = ranking_set.get("season", year)
            ranking_week = ranking_set.get("week")
            polls = ranking_set.get("polls", [])
            
            for poll in polls:
                if poll.get("poll") == "Playoff Committee Rankings":
                    rankings = poll.get("ranks", [])
                    for rank_entry in rankings:
                        records.append({
                            "season": season,
                            "week": ranking_week,
                            "team_id": rank_entry.get("school"),
                            "team": rank_entry.get("school"),
                            "rank": rank_entry.get("rank")
                        })
        
        if not records:
            return pd.DataFrame(columns=["season", "week", "team_id", "team", "rank"])
        
        return pd.DataFrame(records)
    
    def fetch_games(
        self, 
        year: int, 
        season_type: str = "regular",
        week: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch game results for a given year.
        
        Args:
            year: Season year
            season_type: 'regular' or 'postseason'
            week: Optional week number to filter
            
        Returns:
            DataFrame with game results
        """
        params = {
            "year": year,
            "seasonType": season_type
        }
        
        if week is not None:
            params["week"] = week
        
        data = self._make_request("/games", params)
        
        if not data:
            return pd.DataFrame()
        
        # Convert to DataFrame and standardize columns
        df = pd.DataFrame(data)
        
        # Select and rename relevant columns
        columns_map = {
            "id": "game_id",
            "season": "season",
            "week": "week",
            "seasonType": "season_type",
            "startDate": "date",
            "homeTeam": "home_team",
            "awayTeam": "away_team",
            "homePoints": "home_score",
            "awayPoints": "away_score",
            "homeConference": "home_conference",
            "awayConference": "away_conference",
            "neutralSite": "neutral_site"
        }
        
        available_cols = {k: v for k, v in columns_map.items() if k in df.columns}
        df = df.rename(columns=available_cols)
        
        # Add computed columns
        df["winner"] = df.apply(
            lambda row: row["home_team"] if row.get("home_score", 0) > row.get("away_score", 0)
            else row["away_team"] if row.get("away_score", 0) > row.get("home_score", 0)
            else None,
            axis=1
        )
        
        return df
    
    def fetch_teams(self, year: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch team metadata including conference affiliations.
        
        Args:
            year: Optional year to filter teams
            
        Returns:
            DataFrame with team information
        """
        params = {}
        if year:
            params["year"] = year
        
        data = self._make_request("/teams", params)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Select relevant columns
        columns_map = {
            "id": "team_id",
            "school": "team_name",
            "mascot": "mascot",
            "conference": "conference",
            "division": "division"
        }
        
        available_cols = {k: v for k, v in columns_map.items() if k in df.columns}
        df = df.rename(columns=available_cols)
        
        # Add season if provided
        if year:
            df["season"] = year
        
        return df
    
    def fetch_conference_champions(self, year: int) -> pd.DataFrame:
        """
        Fetch conference championship results.
        
        Note: CFBD API may not have a direct endpoint for this.
        This function attempts to identify conference championship games
        and determine winners from game results.
        
        Args:
            year: Season year
            
        Returns:
            DataFrame with conference champions
        """
        # Fetch postseason games (conference championships are typically in postseason)
        games_df = self.fetch_games(year, season_type="postseason")
        
        # Filter for conference championship games
        # These typically have "Championship" in the name or are in week 15-16
        ccg_games = games_df[
            (games_df["week"] >= 15) & 
            (games_df["week"] <= 16) &
            (games_df["home_conference"] == games_df["away_conference"])
        ].copy()
        
        champions = []
        for _, game in ccg_games.iterrows():
            if pd.notna(game.get("winner")):
                conference = game.get("home_conference")
                if pd.notna(conference):
                    champions.append({
                        "season": year,
                        "conference": conference,
                        "champion_team": game["winner"]
                    })
        
        if not champions:
            return pd.DataFrame(columns=["season", "conference", "champion_team"])
        
        return pd.DataFrame(champions)
    
    def fetch_all_historical_data(
        self, 
        start_year: int = 2014, 
        end_year: int = 2023
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all historical data for multiple seasons.
        
        Args:
            start_year: First season to fetch
            end_year: Last season to fetch
            
        Returns:
            Dictionary with keys: 'rankings', 'games', 'teams', 'champions'
        """
        all_rankings = []
        all_games = []
        all_teams = []
        all_champions = []
        
        years = range(start_year, end_year + 1)
        
        print(f"Fetching data for seasons {start_year}-{end_year}...")
        
        for year in tqdm(years, desc="Fetching seasons"):
            # Fetch rankings
            try:
                rankings = self.fetch_rankings(year)
                if not rankings.empty:
                    all_rankings.append(rankings)
            except Exception as e:
                print(f"Warning: Failed to fetch rankings for {year}: {e}")
            
            # Fetch games
            try:
                games = self.fetch_games(year, season_type="regular")
                if not games.empty:
                    all_games.append(games)
                
                # Also fetch postseason games
                post_games = self.fetch_games(year, season_type="postseason")
                if not post_games.empty:
                    all_games.append(post_games)
            except Exception as e:
                print(f"Warning: Failed to fetch games for {year}: {e}")
            
            # Fetch teams
            try:
                teams = self.fetch_teams(year)
                if not teams.empty:
                    all_teams.append(teams)
            except Exception as e:
                print(f"Warning: Failed to fetch teams for {year}: {e}")
            
            # Fetch champions
            try:
                champions = self.fetch_conference_champions(year)
                if not champions.empty:
                    all_champions.append(champions)
            except Exception as e:
                print(f"Warning: Failed to fetch champions for {year}: {e}")
        
        # Combine all dataframes
        result = {
            "rankings": pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame(),
            "games": pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame(),
            "teams": pd.concat(all_teams, ignore_index=True) if all_teams else pd.DataFrame(),
            "champions": pd.concat(all_champions, ignore_index=True) if all_champions else pd.DataFrame()
        }
        
        return result


if __name__ == "__main__":
    # Example usage
    fetcher = CFBDFetcher()
    
    # Test fetching a single year
    print("Testing data fetching...")
    rankings = fetcher.fetch_rankings(2023, week=15)
    print(f"Fetched {len(rankings)} ranking entries")
    print(rankings.head())

