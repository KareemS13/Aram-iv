"""
model.py — LightGBM training, evaluation, and persistence for NO2 forecasting.

Uses a strictly chronological train/val/test split to prevent data leakage.
"""

import pickle
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    FEATURES_CSV,
    FIGURES_DIR,
    LGBM_PARAMS,
    MODEL_PATH,
    MODELS_DIR,
    TRAIN_FRAC,
    VAL_FRAC,
    get_logger,
)
from features import FeatureEngineer

logger = get_logger(__name__)


class NO2Forecaster:
    """
    Train, evaluate, and persist a LightGBM model for next-day NO2 forecasting.

    Parameters
    ----------
    features_csv : Path
        Full feature matrix produced by FeatureEngineer.
    model_path : Path
        Destination for the serialised model (pickle).
    figures_dir : Path
        Directory for evaluation plots.
    """

    def __init__(
        self,
        features_csv: Path = FEATURES_CSV,
        model_path: Path = MODEL_PATH,
        figures_dir: Path = FIGURES_DIR,
    ) -> None:
        self.features_csv = Path(features_csv)
        self.model_path = Path(model_path)
        self.figures_dir = Path(figures_dir)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Data splitting ────────────────────────────────────────────────────────

    def _time_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split df into train/val/test preserving temporal order.

        Proportions: TRAIN_FRAC=0.70, VAL_FRAC=0.15, test=0.15.
        No shuffling — strictly chronological.
        """
        n = len(df)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        train = df.iloc[:n_train]
        val = df.iloc[n_train : n_train + n_val]
        test = df.iloc[n_train + n_val :]
        logger.info(
            "Split: train=%d, val=%d, test=%d (total=%d)",
            len(train), len(val), len(test), n,
        )
        return train, val, test

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self) -> lgb.LGBMRegressor:
        """
        Load features, split, and train LightGBM with early stopping on val MAE.

        Saves the fitted model to self.model_path.
        Returns the fitted model.
        """
        df = pd.read_csv(self.features_csv, index_col="date", parse_dates=True)
        feature_cols = FeatureEngineer.feature_columns()
        available = [c for c in feature_cols if c in df.columns]

        train_df, val_df, _ = self._time_split(df)

        X_train = train_df[available]
        y_train = train_df["no2_next"]
        X_val = val_df[available]
        y_val = val_df["no2_next"]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )

        logger.info(
            "Training complete. Best iteration: %d", model.best_iteration_
        )

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as fh:
            pickle.dump({"model": model, "feature_cols": available}, fh)
        logger.info("Model saved → %s", self.model_path)

        return model

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, model: lgb.LGBMRegressor | None = None) -> dict:
        """
        Compute MAE, RMSE, R² on the held-out test set.

        If model is None, loads from self.model_path.
        Saves prediction and feature-importance plots.

        Returns dict: {test_mae, test_rmse, test_r2} in mol/m².
        """
        if model is None:
            model, available = self.load_model()
        else:
            df_tmp = pd.read_csv(self.features_csv, index_col="date", parse_dates=True)
            feature_cols = FeatureEngineer.feature_columns()
            available = [c for c in feature_cols if c in df_tmp.columns]

        df = pd.read_csv(self.features_csv, index_col="date", parse_dates=True)

        _, _, test_df = self._time_split(df)
        X_test = test_df[available]
        y_test = test_df["no2_next"]

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)

        # Display in µmol/m² for readability
        logger.info(
            "Test MAE=%.4f µmol/m²  RMSE=%.4f µmol/m²  R²=%.4f",
            mae, rmse, r2,
        )
        print(
            f"\nTest set results:\n"
            f"  MAE  = {mae:.4f} µmol/m²\n"
            f"  RMSE = {rmse:.4f} µmol/m²\n"
            f"  R²   = {r2:.4f}\n"
        )

        self._plot_predictions(y_test, y_pred)
        self._plot_feature_importance(model, available)

        return {"test_mae": mae, "test_rmse": rmse, "test_r2": r2}

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _plot_predictions(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        split_name: str = "test",
    ) -> None:
        """Save a time-series plot of predicted vs. actual NO2."""
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(y_true.index, y_true.values, label="Actual", linewidth=1.2)
        ax.plot(y_true.index, y_pred, label="Predicted", linewidth=1.0, alpha=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Tropospheric NO\u2082 (µmol/m²)")
        ax.set_title(f"NO\u2082 Forecast vs Actual — {split_name} set")
        ax.legend()
        fig.tight_layout()
        dest = self.figures_dir / f"predictions_{split_name}.png"
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        logger.info("Saved prediction plot → %s", dest)

    def _plot_feature_importance(
        self, model: lgb.LGBMRegressor, feature_names: list[str]
    ) -> None:
        """Save a horizontal bar chart of the top-20 feature importances."""
        importances = model.feature_importances_
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
        names, scores = zip(*pairs)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(list(reversed(names)), list(reversed(scores)))
        ax.set_xlabel("Importance (split count)")
        ax.set_title("Top-20 Feature Importances")
        fig.tight_layout()
        dest = self.figures_dir / "feature_importance.png"
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        logger.info("Saved feature importance plot → %s", dest)

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_model(self) -> tuple[lgb.LGBMRegressor, list[str]]:
        """Load and return (model, feature_cols) from self.model_path."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No model found at {self.model_path}. Run 'train' first."
            )
        with open(self.model_path, "rb") as fh:
            payload = pickle.load(fh)
        # Support both old format (model only) and new format (dict)
        if isinstance(payload, dict):
            model = payload["model"]
            feature_cols = payload["feature_cols"]
        else:
            model = payload
            feature_cols = FeatureEngineer.feature_columns()
        logger.info("Model loaded from %s", self.model_path)
        return model, feature_cols
