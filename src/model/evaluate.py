"""
Model evaluation metrics.

This module provides functions to evaluate model performance
using ranking-specific metrics.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from scipy.stats import kendalltau, spearmanr


def kendall_tau(predicted_ranks: pd.Series, actual_ranks: pd.Series) -> float:
    """
    Compute Kendall's tau correlation between predicted and actual ranks.
    
    Args:
        predicted_ranks: Series of predicted ranks
        actual_ranks: Series of actual ranks
        
    Returns:
        Kendall's tau coefficient (-1 to 1, higher is better)
    """
    # Align by team/index
    aligned = pd.DataFrame({
        "predicted": predicted_ranks,
        "actual": actual_ranks
    }).dropna()
    
    if len(aligned) < 2:
        return 0.0
    
    tau, _ = kendalltau(aligned["predicted"], aligned["actual"])
    return float(tau) if not np.isnan(tau) else 0.0


def spearman_correlation(predicted_ranks: pd.Series, actual_ranks: pd.Series) -> float:
    """
    Compute Spearman rank correlation.
    
    Args:
        predicted_ranks: Series of predicted ranks
        actual_ranks: Series of actual ranks
        
    Returns:
        Spearman correlation coefficient (-1 to 1, higher is better)
    """
    aligned = pd.DataFrame({
        "predicted": predicted_ranks,
        "actual": actual_ranks
    }).dropna()
    
    if len(aligned) < 2:
        return 0.0
    
    corr, _ = spearmanr(aligned["predicted"], aligned["actual"])
    return float(corr) if not np.isnan(corr) else 0.0


def top_n_accuracy(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    n: int = 4,
    team_col: str = "team"
) -> float:
    """
    Compute accuracy of top N predictions.
    
    Measures how many of the actual top N teams are in the predicted top N.
    
    Args:
        predicted: DataFrame with predicted rankings (sorted by rank)
        actual: DataFrame with actual rankings (sorted by rank)
        n: Number of top teams to check
        team_col: Column name for team identifier
        
    Returns:
        Accuracy as fraction (0 to 1)
    """
    pred_top_n = set(predicted.head(n)[team_col].values)
    actual_top_n = set(actual.head(n)[team_col].values)
    
    if len(actual_top_n) == 0:
        return 0.0
    
    correct = len(pred_top_n & actual_top_n)
    return correct / len(actual_top_n)


def mean_rank_error(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    team_col: str = "team",
    rank_col: str = "rank"
) -> float:
    """
    Compute mean absolute rank error.
    
    Args:
        predicted: DataFrame with predicted rankings
        actual: DataFrame with actual rankings
        team_col: Column name for team identifier
        rank_col: Column name for rank
        
    Returns:
        Mean absolute error in rank positions
    """
    # Merge on team
    merged = pd.merge(
        predicted[[team_col, rank_col]],
        actual[[team_col, rank_col]],
        on=team_col,
        suffixes=("_pred", "_actual")
    )
    
    if merged.empty:
        return float('inf')
    
    errors = np.abs(merged[f"{rank_col}_pred"] - merged[f"{rank_col}_actual"])
    return float(errors.mean())


def evaluate_model(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    team_col: str = "team",
    rank_col: str = "rank"
) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        predicted: DataFrame with predicted rankings
        actual: DataFrame with actual rankings
        team_col: Column name for team identifier
        rank_col: Column name for rank
        
    Returns:
        Dictionary with all metrics
    """
    # Ensure sorted by rank
    pred_sorted = predicted.sort_values(rank_col).reset_index(drop=True)
    actual_sorted = actual.sort_values(rank_col).reset_index(drop=True)
    
    # Align by team for rank-based metrics
    merged = pd.merge(
        pred_sorted[[team_col, rank_col]],
        actual_sorted[[team_col, rank_col]],
        on=team_col,
        suffixes=("_pred", "_actual")
    )
    
    metrics = {
        "kendall_tau": kendall_tau(merged[f"{rank_col}_pred"], merged[f"{rank_col}_actual"]),
        "spearman_correlation": spearman_correlation(merged[f"{rank_col}_pred"], merged[f"{rank_col}_actual"]),
        "mean_rank_error": mean_rank_error(pred_sorted, actual_sorted, team_col, rank_col),
        "top_4_accuracy": top_n_accuracy(pred_sorted, actual_sorted, n=4, team_col=team_col),
        "top_12_accuracy": top_n_accuracy(pred_sorted, actual_sorted, n=12, team_col=team_col),
        "top_25_accuracy": top_n_accuracy(pred_sorted, actual_sorted, n=25, team_col=team_col)
    }
    
    return metrics

