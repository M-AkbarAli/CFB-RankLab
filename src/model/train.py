"""
Model training script.

This module trains the CFP ranking prediction model using historical data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings

# Try to import XGBoost, fall back to LightGBM if it fails
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError as e:
    XGBOOST_AVAILABLE = False
    warnings.warn(f"XGBoost not available: {e}. Will use LightGBM instead.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError as e:
    LIGHTGBM_AVAILABLE = False
    if not XGBOOST_AVAILABLE:
        raise ImportError("Neither XGBoost nor LightGBM are available. Please install at least one.")

from ..features.compute import compute_features_for_all_weeks
from .evaluate import evaluate_model


def prepare_training_data(
    games_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    champions_df: Optional[pd.DataFrame] = None,
    seasons: Optional[list] = None
) -> pd.DataFrame:
    """
    Prepare training dataset by computing features for all historical weeks.
    
    Args:
        games_df: Processed games DataFrame
        teams_df: Processed teams DataFrame
        rankings_df: Processed rankings DataFrame
        champions_df: Optional conference champions DataFrame
        seasons: Optional list of seasons to include (default: all in rankings_df)
        
    Returns:
        DataFrame with features and target ranks
    """
    if seasons is None:
        seasons = sorted(rankings_df["season"].unique().tolist())
    
    print(f"Computing features for seasons {seasons}...")
    
    features_df = compute_features_for_all_weeks(
        seasons=seasons,
        games_df=games_df,
        teams_df=teams_df,
        rankings_df=rankings_df,
        champions_df=champions_df
    )
    
    return features_df


def encode_categorical_features(features_df: pd.DataFrame) -> tuple:
    """
    Encode categorical features for model training.
    
    Args:
        features_df: Features DataFrame
        
    Returns:
        Tuple of (encoded_df, label_encoders_dict)
    """
    df = features_df.copy()
    encoders = {}
    
    # Encode conference
    if "conference" in df.columns:
        le = LabelEncoder()
        df["conference_encoded"] = le.fit_transform(df["conference"].astype(str))
        encoders["conference"] = le
        df = df.drop(columns=["conference"])
    
    return df, encoders


def train_model(
    training_data: pd.DataFrame,
    validation_season: Optional[int] = None,
    model_type: str = "xgboost",
    hyperparameters: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Train the CFP ranking prediction model.
    
    Args:
        training_data: DataFrame with features and target_rank
        validation_season: Optional season to hold out for validation
        model_type: "xgboost" or "lightgbm"
        hyperparameters: Optional hyperparameter dict
        
    Returns:
        Tuple of (trained_model, encoders, evaluation_metrics)
    """
    # Encode categorical features
    data_encoded, encoders = encode_categorical_features(training_data)
    
    # Prepare features and target
    exclude_cols = {"team", "season", "week", "target_rank"}
    feature_cols = [col for col in data_encoded.columns if col not in exclude_cols]
    
    X = data_encoded[feature_cols].fillna(0)
    y = data_encoded["target_rank"].fillna(26)  # Unranked = 26
    
    # Train/validation split
    if validation_season is not None:
        # Temporal split: train on earlier seasons, validate on later
        train_mask = data_encoded["season"] < validation_season
        val_mask = data_encoded["season"] >= validation_season
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]
    else:
        # Random split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Default hyperparameters
    if hyperparameters is None:
        if model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                print("Warning: XGBoost not available, switching to LightGBM")
                model_type = "lightgbm"
            hyperparameters = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "objective": "reg:squarederror"
            }
        else:  # lightgbm
            hyperparameters = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "objective": "regression"
            }
    
    # Train model
    print(f"Training {model_type} model...")
    if model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not available. This is often due to missing OpenMP runtime on macOS. "
                "Try: brew install libomp\n"
                "Or use LightGBM instead by setting model_type='lightgbm'"
            )
        model = xgb.XGBRegressor(**hyperparameters)
    else:
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Please install it: pip install lightgbm")
        model = lgb.LGBMRegressor(**hyperparameters)
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Create DataFrames for evaluation
    train_pred_df = pd.DataFrame({
        "team": data_encoded.loc[train_mask, "team"].values if validation_season else data_encoded.loc[X_train.index, "team"].values,
        "rank": y_pred_train
    }).sort_values("rank").reset_index(drop=True)
    train_pred_df["rank"] = range(1, len(train_pred_df) + 1)
    
    train_actual_df = pd.DataFrame({
        "team": data_encoded.loc[train_mask, "team"].values if validation_season else data_encoded.loc[X_train.index, "team"].values,
        "rank": y_train.values
    }).sort_values("rank").reset_index(drop=True)
    train_actual_df["rank"] = range(1, len(train_actual_df) + 1)
    
    val_pred_df = pd.DataFrame({
        "team": data_encoded.loc[val_mask, "team"].values if validation_season else data_encoded.loc[X_val.index, "team"].values,
        "rank": y_pred_val
    }).sort_values("rank").reset_index(drop=True)
    val_pred_df["rank"] = range(1, len(val_pred_df) + 1)
    
    val_actual_df = pd.DataFrame({
        "team": data_encoded.loc[val_mask, "team"].values if validation_season else data_encoded.loc[X_val.index, "team"].values,
        "rank": y_val.values
    }).sort_values("rank").reset_index(drop=True)
    val_actual_df["rank"] = range(1, len(val_actual_df) + 1)
    
    train_metrics = evaluate_model(train_pred_df, train_actual_df)
    val_metrics = evaluate_model(val_pred_df, val_actual_df)
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Store feature columns for later use
    model.feature_names_ = feature_cols
    
    return model, encoders, {"train": train_metrics, "validation": val_metrics}


def save_model(
    model: object,
    encoders: dict,
    model_path: Optional[Path] = None,
    metadata: Optional[dict] = None
):
    """
    Save trained model and encoders.
    
    Args:
        model: Trained model object
        encoders: Dictionary of label encoders
        model_path: Path to save model
        metadata: Optional metadata to save
    """
    if model_path is None:
        model_path = Path(__file__).parent.parent.parent / "data" / "models" / "cfp_predictor.pkl"
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model and encoders together
    save_dict = {
        "model": model,
        "encoders": encoders,
        "metadata": metadata or {}
    }
    
    joblib.dump(save_dict, model_path)
    print(f"Model saved to {model_path}")


def main():
    """
    Main training script.
    
    This can be run as a standalone script to train the model.
    """
    from ..data.fetcher import CFBDFetcher
    from ..data.processor import DataProcessor
    
    # Load data
    print("Loading data...")
    fetcher = CFBDFetcher()
    processor = DataProcessor()
    
    # Fetch historical data (2014-2023)
    print("Fetching historical data...")
    historical_data = fetcher.fetch_all_historical_data(2014, 2023)
    
    # Process data
    print("Processing data...")
    print(f"  - Processing {len(historical_data['games'])} game records...")
    games_df = processor.process_games(historical_data["games"])
    print(f"    ✓ Created {len(games_df)} team-game records")
    
    print(f"  - Processing {len(historical_data['teams'])} team records...")
    teams_df = processor.process_teams(historical_data["teams"])
    print(f"    ✓ Processed {len(teams_df)} teams")
    
    print(f"  - Processing {len(historical_data['rankings'])} ranking records...")
    rankings_df = processor.process_rankings(historical_data["rankings"])
    print(f"    ✓ Processed {len(rankings_df)} rankings")
    
    print(f"  - Processing {len(historical_data.get('champions', pd.DataFrame()))} champion records...")
    champions_df = processor.process_champions(historical_data.get("champions", pd.DataFrame()))
    print(f"    ✓ Processed {len(champions_df)} champions")
    
    # Prepare training data
    print("\nPreparing training data...")
    print("  - Computing features for each team-game...")
    training_data = prepare_training_data(
        games_df=games_df,
        teams_df=teams_df,
        rankings_df=rankings_df,
        champions_df=champions_df,
        seasons=list(range(2014, 2024))
    )
    print("  ✓ Feature computation complete")
    
    if training_data.empty:
        print("Error: No training data generated!")
        return
    
    print(f"Training data shape: {training_data.shape}")
    
    # Train model (temporal split: train on 2014-2020, validate on 2021-2023)
    # Try XGBoost first, fall back to LightGBM if XGBoost fails
    model_type = "xgboost" if XGBOOST_AVAILABLE else "lightgbm"
    print(f"Using {model_type} for training...")
    
    try:
        model, encoders, metrics = train_model(
            training_data=training_data,
            validation_season=2021,
            model_type=model_type
        )
    except Exception as e:
        if model_type == "xgboost" and LIGHTGBM_AVAILABLE:
            print(f"\nXGBoost failed: {e}")
            print("Falling back to LightGBM...")
            model, encoders, metrics = train_model(
                training_data=training_data,
                validation_season=2021,
                model_type="lightgbm"
            )
        else:
            raise
    
    # Save model
    save_model(
        model=model,
        encoders=encoders,
        metadata={
            "training_seasons": list(range(2014, 2021)),
            "validation_seasons": list(range(2021, 2024)),
            "metrics": metrics
        }
    )
    
    print("\nModel training complete!")


if __name__ == "__main__":
    main()

