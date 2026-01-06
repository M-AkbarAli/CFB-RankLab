"""
Unit tests for feature computation.
"""

import pytest
import pandas as pd
import numpy as np

from src.features.utils import (
    compute_record_features,
    compute_sos,
    compute_quality_wins,
    compute_conference_features,
    compute_momentum_features
)


@pytest.fixture
def sample_games():
    """Sample games DataFrame for testing."""
    return pd.DataFrame({
        "season": [2023, 2023, 2023],
        "week": [1, 2, 3],
        "team": ["Team A", "Team A", "Team B"],
        "opponent": ["Team B", "Team C", "Team A"],
        "team_won": [True, True, False],
        "team_score": [35, 28, 14],
        "opp_score": [14, 21, 35],
        "is_conference_game": [True, False, True]
    })


@pytest.fixture
def sample_teams():
    """Sample teams DataFrame for testing."""
    return pd.DataFrame({
        "team_id": ["Team A", "Team B", "Team C"],
        "season": [2023, 2023, 2023],
        "conference": ["SEC", "Big Ten", "ACC"]
    })


def test_compute_record_features(sample_games):
    """Test record feature computation."""
    features = compute_record_features(
        sample_games, "Team A", 2023, 3
    )
    
    assert features["wins"] == 2
    assert features["losses"] == 0
    assert features["win_pct"] == 1.0


def test_compute_sos(sample_games):
    """Test SOS computation."""
    team_records = pd.DataFrame({
        "team": ["Team B", "Team C"],
        "season": [2023, 2023],
        "as_of_week": [1, 2],
        "wins": [0, 1],
        "games_played": [1, 1],
        "win_pct": [0.0, 1.0]
    })
    
    features = compute_sos(
        sample_games, "Team A", 2023, 3, team_records
    )
    
    assert "sos_score" in features
    assert features["sos_score"] >= 0


def test_compute_conference_features(sample_teams):
    """Test conference feature computation."""
    features = compute_conference_features(
        sample_teams, "Team A", 2023
    )
    
    assert features["conference"] == "SEC"
    assert features["is_power5"] == True


def test_compute_momentum_features(sample_games):
    """Test momentum feature computation."""
    features = compute_momentum_features(
        sample_games, "Team A", 2023, 3
    )
    
    assert "current_win_streak" in features
    assert features["current_win_streak"] >= 0

