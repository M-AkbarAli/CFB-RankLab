"""
Unit tests for playoff selection logic.
"""

import pytest
import pandas as pd

from src.simulation.playoff import (
    select_playoff_teams,
    assign_seeds,
    generate_bracket
)


@pytest.fixture
def sample_rankings():
    """Sample rankings DataFrame."""
    teams = [f"Team {i}" for i in range(1, 26)]
    return pd.DataFrame({
        "team": teams,
        "predicted_rank": range(1, 26)
    })


@pytest.fixture
def sample_champions():
    """Sample conference champions."""
    return {
        "SEC": "Team 1",
        "Big Ten": "Team 2",
        "Big 12": "Team 3",
        "ACC": "Team 4",
        "Pac-12": "Team 5"
    }


def test_select_playoff_teams_12team(sample_rankings, sample_champions):
    """Test 12-team playoff selection."""
    playoff = select_playoff_teams(
        sample_rankings, sample_champions, format="12team"
    )
    
    assert len(playoff) == 12
    assert "seed" in playoff.columns
    assert "is_auto_bid" in playoff.columns


def test_assign_seeds(sample_rankings, sample_champions):
    """Test seed assignment."""
    playoff = select_playoff_teams(
        sample_rankings, sample_champions, format="12team"
    )
    
    # Top 4 should be conference champions
    top_4 = playoff[playoff["seed"] <= 4]
    assert all(top_4["is_auto_bid"])


def test_generate_bracket(sample_rankings, sample_champions):
    """Test bracket generation."""
    playoff = select_playoff_teams(
        sample_rankings, sample_champions, format="12team"
    )
    
    bracket = generate_bracket(playoff)
    
    assert not bracket.empty
    assert "round" in bracket.columns

