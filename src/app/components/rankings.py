"""
Rankings display component for Streamlit app.
"""

import pandas as pd
import streamlit as st
from typing import Optional


def display_rankings(
    rankings_df: pd.DataFrame,
    title: str = "CFP Rankings",
    show_features: bool = False,
    highlight_playoff: bool = False,
    playoff_teams: Optional[set] = None,
    baseline_rankings: Optional[pd.DataFrame] = None
):
    """
    Display CFP rankings in a formatted table.
    
    Args:
        rankings_df: DataFrame with rankings (team, predicted_rank, predicted_score)
        title: Title for the display
        show_features: Whether to show additional feature columns
        highlight_playoff: Whether to highlight playoff teams
        playoff_teams: Optional set of playoff team names
        baseline_rankings: Optional baseline for comparison
    """
    st.subheader(title)
    
    if rankings_df.empty:
        st.warning("No rankings available.")
        return
    
    # Prepare display DataFrame
    display_df = rankings_df.copy()
    
    # Ensure sorted by rank
    if "predicted_rank" in display_df.columns:
        display_df = display_df.sort_values("predicted_rank").reset_index(drop=True)
        rank_col = "predicted_rank"
    elif "rank" in display_df.columns:
        display_df = display_df.sort_values("rank").reset_index(drop=True)
        rank_col = "rank"
    else:
        st.error("No rank column found in rankings DataFrame.")
        return
    
    # Select columns to display
    display_cols = ["team", rank_col]
    
    if "predicted_score" in display_df.columns:
        display_cols.append("predicted_score")
    
    if show_features:
        # Add any feature columns if present
        feature_cols = ["wins", "losses", "sos_rank", "wins_vs_winning_teams"]
        for col in feature_cols:
            if col in display_df.columns:
                display_cols.append(col)
    
    display_df = display_df[display_cols].head(25)  # Top 25
    
    # Add comparison if baseline provided
    if baseline_rankings is not None:
        baseline_dict = dict(zip(
            baseline_rankings["team"],
            baseline_rankings.get("predicted_rank", baseline_rankings.get("rank", []))
        ))
        
        display_df["baseline_rank"] = display_df["team"].map(baseline_dict)
        display_df["rank_change"] = (
            display_df["baseline_rank"] - display_df[rank_col]
        ).fillna(0)
        
        # Format rank change
        def format_change(change):
            if pd.isna(change) or change == 0:
                return "—"
            elif change > 0:
                return f"↑{int(change)}"
            else:
                return f"↓{int(abs(change))}"
        
        display_df["change"] = display_df["rank_change"].apply(format_change)
        display_cols.extend(["baseline_rank", "change"])
    
    # Rename columns for display
    rename_map = {
        "team": "Team",
        "predicted_rank": "Rank",
        "rank": "Rank",
        "predicted_score": "CFP Score",
        "wins": "Wins",
        "losses": "Losses",
        "sos_rank": "SOS Rank",
        "wins_vs_winning_teams": "Quality Wins",
        "baseline_rank": "Prev Rank",
        "change": "Change"
    }
    
    display_df = display_df.rename(columns=rename_map)
    display_cols = [rename_map.get(col, col) for col in display_cols if col in display_df.columns]
    
    # Style the DataFrame
    styled_df = display_df[display_cols].copy()
    
    # Highlight playoff teams
    if highlight_playoff and playoff_teams:
        def highlight_row(row):
            if row["Team"] in playoff_teams:
                return ['background-color: #90EE90'] * len(row)
            return [''] * len(row)
        
        styled_df = styled_df.style.apply(highlight_row, axis=1)
    
    # Display
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Show summary stats
    if highlight_playoff and playoff_teams:
        st.caption(f"🟢 Highlighted: {len(playoff_teams)} playoff teams")


def display_rankings_comparison(
    baseline: pd.DataFrame,
    scenario: pd.DataFrame,
    title: str = "Rankings Comparison"
):
    """
    Display side-by-side comparison of baseline vs scenario rankings.
    
    Args:
        baseline: Baseline rankings DataFrame
        scenario: Scenario rankings DataFrame
        title: Title for display
    """
    st.subheader(title)
    
    # Merge on team
    comparison = pd.merge(
        baseline[["team", "predicted_rank"]].rename(columns={"predicted_rank": "baseline_rank"}),
        scenario[["team", "predicted_rank"]].rename(columns={"predicted_rank": "scenario_rank"}),
        on="team",
        how="outer"
    )
    
    comparison["rank_change"] = comparison["baseline_rank"] - comparison["scenario_rank"]
    comparison = comparison.sort_values("scenario_rank")
    
    # Format for display
    comparison_display = comparison[["team", "baseline_rank", "scenario_rank", "rank_change"]].copy()
    comparison_display.columns = ["Team", "Baseline Rank", "Scenario Rank", "Change"]
    
    st.dataframe(comparison_display, use_container_width=True, hide_index=True)

