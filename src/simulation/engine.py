"""
Scenario simulation engine.

This module handles "what-if" scenario simulations by updating game results
and recalculating features to generate new predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

from ..features.compute import compute_features
from ..model.predict import predict_rankings, load_model


class SimulationEngine:
    """
    Engine for simulating CFP scenarios.
    
    Takes base season data and user-specified game outcomes,
    then generates updated rankings and playoff projections.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize simulation engine.
        
        Args:
            model_path: Path to trained model. If None, uses default location.
        """
        self.model = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            self.model = load_model(self.model_path)
        except FileNotFoundError:
            print("Warning: Model not found. Simulation will not work until model is trained.")
            self.model = None
    
    def simulate_scenario(
        self,
        base_games_df: pd.DataFrame,
        base_teams_df: pd.DataFrame,
        game_outcomes: Dict[str, str],
        target_week: int,
        season: int,
        base_rankings_df: Optional[pd.DataFrame] = None,
        champions_df: Optional[pd.DataFrame] = None,
        previous_rankings_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Simulate a scenario with modified game outcomes.
        
        Args:
            base_games_df: Original games DataFrame
            game_outcomes: Dict mapping game_id to winner team name
            target_week: Week to project rankings for
            season: Season year
            base_teams_df: Teams DataFrame
            base_rankings_df: Optional baseline rankings for comparison
            champions_df: Optional conference champions DataFrame
            previous_rankings_df: Optional previous week's rankings
            
        Returns:
            DataFrame with predicted rankings (team, predicted_rank, predicted_score)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot generate predictions.")
        
        # Create a copy of games DataFrame
        updated_games = base_games_df.copy()
        
        # Update game outcomes
        updated_games = self._update_game_outcomes(updated_games, game_outcomes, season)
        
        # Recompute features for all teams
        features_df = compute_features(
            season=season,
            week=target_week,
            games_df=updated_games,
            teams_df=base_teams_df,
            rankings_df=None,  # No target for prediction
            champions_df=champions_df,
            previous_rankings_df=previous_rankings_df,
        )
        
        if features_df.empty:
            return pd.DataFrame()
        
        # Generate predictions
        predictions = predict_rankings(features_df, model=self.model)
        
        return predictions
    
    def _update_game_outcomes(
        self,
        games_df: pd.DataFrame,
        game_outcomes: Dict[str, str],
        season: int
    ) -> pd.DataFrame:
        """
        Update game results based on user-specified outcomes.
        
        Args:
            games_df: Games DataFrame
            game_outcomes: Dict mapping game_id to winner
            season: Season year
            
        Returns:
            Updated games DataFrame
        """
        updated = games_df.copy()
        
        for game_id, winner in game_outcomes.items():
            # Find games matching this game_id
            game_mask = (
                (updated["game_id"] == game_id) |
                (updated.get("game_id", pd.Series()) == game_id)
            )
            
            # If game_id column doesn't exist, try to match by season/week/teams
            if not game_mask.any() and "game_id" not in updated.columns:
                # Try to construct game_id from other columns
                # This is a fallback if game_id wasn't in original data
                continue
            
            games_to_update = updated[game_mask]
            
            for idx in games_to_update.index:
                game = updated.loc[idx]
                
                # Determine if the team in this row won or lost
                if game["team"] == winner:
                    # This team won
                    updated.at[idx, "team_won"] = True
                    updated.at[idx, "team_score"] = max(
                        updated.at[idx, "team_score"] if pd.notna(updated.at[idx, "team_score"]) else 0,
                        updated.at[idx, "opp_score"] if pd.notna(updated.at[idx, "opp_score"]) else 0
                    ) + 1  # Ensure winner has higher score
                    if pd.notna(updated.at[idx, "opp_score"]):
                        updated.at[idx, "opp_score"] = updated.at[idx, "team_score"] - 1
                elif game["opponent"] == winner:
                    # Opponent won (this team lost)
                    updated.at[idx, "team_won"] = False
                    updated.at[idx, "opp_score"] = max(
                        updated.at[idx, "team_score"] if pd.notna(updated.at[idx, "team_score"]) else 0,
                        updated.at[idx, "opp_score"] if pd.notna(updated.at[idx, "opp_score"]) else 0
                    ) + 1
                    if pd.notna(updated.at[idx, "team_score"]):
                        updated.at[idx, "team_score"] = updated.at[idx, "opp_score"] - 1
        
        return updated
    
    def compare_scenarios(
        self,
        baseline_rankings: pd.DataFrame,
        scenario_rankings: pd.DataFrame,
        team_col: str = "team",
        rank_col: str = "predicted_rank"
    ) -> pd.DataFrame:
        """
        Compare baseline vs scenario rankings to show changes.
        
        Args:
            baseline_rankings: Baseline predicted rankings
            scenario_rankings: Scenario predicted rankings
            team_col: Column name for team
            rank_col: Column name for rank
            
        Returns:
            DataFrame with comparison (team, baseline_rank, scenario_rank, rank_change)
        """
        baseline = baseline_rankings[[team_col, rank_col]].copy()
        baseline.columns = [team_col, "baseline_rank"]
        
        scenario = scenario_rankings[[team_col, rank_col]].copy()
        scenario.columns = [team_col, "scenario_rank"]
        
        comparison = pd.merge(baseline, scenario, on=team_col, how="outer")
        comparison["rank_change"] = comparison["baseline_rank"] - comparison["scenario_rank"]
        # Positive change = moved up, negative = moved down
        
        return comparison.sort_values("scenario_rank")


def simulate_scenario(
    base_games_df: pd.DataFrame,
    base_teams_df: pd.DataFrame,
    game_outcomes: Dict[str, str],
    target_week: int,
    season: int,
    model_path: Optional[Path] = None,
    champions_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Convenience function to simulate a scenario.
    
    Args:
        base_games_df: Original games DataFrame
        base_teams_df: Teams DataFrame
        game_outcomes: Dict mapping game_id to winner
        target_week: Week to project
        season: Season year
        model_path: Optional path to model
        champions_df: Optional conference champions
        
    Returns:
        Predicted rankings DataFrame
    """
    engine = SimulationEngine(model_path=model_path)
    return engine.simulate_scenario(
        base_games_df=base_games_df,
        base_teams_df=base_teams_df,
        game_outcomes=game_outcomes,
        target_week=target_week,
        season=season,
        champions_df=champions_df
    )

