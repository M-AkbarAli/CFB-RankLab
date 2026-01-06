"""
Model prediction functions.

This module provides functions to load trained models and generate predictions.
"""

import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
import joblib


def load_model(model_path: Optional[Path] = None) -> object:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to saved model. If None, uses default location.
        
    Returns:
        Loaded model object (or dict with 'model' key if saved as dict)
    """
    if model_path is None:
        model_path = Path(__file__).parent.parent.parent / "data" / "models" / "cfp_predictor.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    
    loaded = joblib.load(model_path)
    
    # Handle case where model was saved as dict
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"]
    
    return loaded


def predict_rankings(
    features_df: pd.DataFrame,
    model: Optional[object] = None,
    model_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Predict CFP rankings from team features.
    
    Args:
        features_df: DataFrame with feature columns (one row per team)
        model: Optional pre-loaded model. If None, loads from model_path.
        model_path: Path to saved model (used if model is None)
        
    Returns:
        DataFrame with columns: team, predicted_score, predicted_rank
    """
    if model is None:
        model = load_model(model_path)
    
    # Identify feature columns (exclude metadata columns)
    exclude_cols = {"team", "season", "week", "target_rank", "conference"}
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    # Handle categorical columns (conference)
    if "conference" in features_df.columns:
        # One-hot encode conference or use label encoding
        # For simplicity, we'll drop it if model doesn't handle it
        # In practice, the model should be trained with encoded conference
        pass
    
    # Extract features
    X = features_df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Predict scores (lower score = better rank)
    predicted_scores = model.predict(X)
    
    # Create results DataFrame
    results = features_df[["team"]].copy()
    results["predicted_score"] = predicted_scores
    
    # Sort by score to assign ranks (lower = better)
    results = results.sort_values("predicted_score").reset_index(drop=True)
    results["predicted_rank"] = range(1, len(results) + 1)
    
    return results


def predict_top_n(
    features_df: pd.DataFrame,
    n: int = 25,
    model: Optional[object] = None,
    model_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Predict Top N teams.
    
    Args:
        features_df: DataFrame with feature columns
        n: Number of teams to return (default 25)
        model: Optional pre-loaded model
        model_path: Path to saved model
        
    Returns:
        DataFrame with top N teams by predicted rank
    """
    rankings = predict_rankings(features_df, model, model_path)
    return rankings.head(n)

