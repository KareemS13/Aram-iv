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
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURES_CSV,
    FIGURES_DIR,
    LGBM_PARAMS,
    MODEL_PATH,
    MODELS_DIR,
    RF_PARAMS,
    RIDGE_PARAMS,
    TRAIN_FRAC,
    VAL_FRAC,
    XGB_PARAMS,
    get_logger,
)
from features import FeatureEngineer

logger = get_logger(__name__)


class EnsembleModel:
    """
    Weighted ensemble of LightGBM, XGBoost, Random Forest, and Ridge Regression.
    Exposes .predict(X) as a drop-in replacement for any single sklearn-style regressor.
    """

    def __init__(self, models: dict, weights: dict, scaler: StandardScaler | None = None, train_means: pd.Series | None = None) -> None:
        self.models = models        # {"lgbm": ..., "xgb": ..., "rf": ..., "ridge": ...}
        self.weights = weights      # {"lgbm": float, ...} — sum to 1.0
        self.scaler = scaler        # fitted StandardScaler for Ridge
        self.train_means = train_means  # column means for NaN imputation before Ridge

    def predict(self, X) -> np.ndarray:
        total = np.zeros(len(X))
        for name, model in self.models.items():
            if name == "ridge" and self.scaler is not None:
                X_filled = pd.DataFrame(X).fillna(self.train_means)
                X_scaled = self.scaler.transform(X_filled)
                total += self.weights[name] * model.predict(X_scaled)
            else:
                total += self.weights[name] * model.predict(X)
        return total

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return LightGBM feature importances for the existing plot code."""
        return self.models["lgbm"].feature_importances_


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

    # ── LightGBM grid search ──────────────────────────────────────────────────

    def _grid_search_lgbm(self, X_train, y_train, X_val, y_val) -> dict:
        """Find best LightGBM hyperparameters via validation R²."""
        from itertools import product as iproduct
        param_grid = {
            "num_leaves":        [15, 31],
            "min_child_samples": [5, 10, 20],
            "reg_lambda":        [0.05, 0.1, 0.2],
        }
        base_params = {
            k: v for k, v in LGBM_PARAMS.items()
            if k not in param_grid
        }
        best_r2, best_params = -np.inf, {}
        keys, values = zip(*param_grid.items())
        combos = list(iproduct(*values))
        logger.info("LGBM grid search: %d combinations …", len(combos))
        for combo in combos:
            params = dict(zip(keys, combo))
            m = lgb.LGBMRegressor(**base_params, **params)
            m.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            r2 = r2_score(y_val, m.predict(X_val))
            if r2 > best_r2:
                best_r2, best_params = r2, params
        logger.info("LGBM best params: %s  val R²=%.4f", best_params, best_r2)
        return best_params

    # ── XGBoost grid search ───────────────────────────────────────────────────

    def _grid_search_xgb(self, X_train, y_train, X_val, y_val) -> dict:
        """Find best XGBoost hyperparameters via validation R²."""
        from itertools import product as iproduct
        param_grid = {
            "max_depth":        [3, 4, 5, 6],
            "learning_rate":    [0.01, 0.02, 0.05],
            "min_child_weight": [3, 5, 10],
            "subsample":        [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        }
        base_params = {
            k: v for k, v in XGB_PARAMS.items()
            if k not in param_grid and k not in ("early_stopping_rounds", "verbosity")
        }
        best_r2, best_params = -np.inf, {}
        keys, values = zip(*param_grid.items())
        combos = list(iproduct(*values))
        logger.info("XGB grid search: %d combinations …", len(combos))
        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            m = xgb.XGBRegressor(
                **base_params, **params,
                early_stopping_rounds=100, verbosity=0,
            )
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            r2 = r2_score(y_val, m.predict(X_val))
            if r2 > best_r2:
                best_r2, best_params = r2, params
            if (i + 1) % 50 == 0:
                logger.info("  [%d/%d] best val R²=%.4f", i + 1, len(combos), best_r2)
        logger.info("XGB best params: %s  val R²=%.4f", best_params, best_r2)
        return best_params

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self) -> EnsembleModel:
        """
        Train LightGBM, XGBoost, Random Forest, and Ridge Regression.
        Determines ensemble weights from validation R², saves EnsembleModel.
        Returns the fitted EnsembleModel.
        """
        df = pd.read_csv(self.features_csv, index_col="date", parse_dates=True)
        feature_cols = FeatureEngineer.feature_columns()
        available = [c for c in feature_cols if c in df.columns]

        train_df, val_df, _ = self._time_split(df)
        X_train = train_df[available]
        y_train = train_df["no2_next"]
        X_val = val_df[available]
        y_val = val_df["no2_next"]

        # --- LightGBM ---
        lgbm_model = lgb.LGBMRegressor(**LGBM_PARAMS)
        lgbm_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )
        logger.info("LightGBM best iteration: %d", lgbm_model.best_iteration_)

        # --- XGBoost (with grid search) ---
        best_xgb_params = self._grid_search_xgb(X_train, y_train, X_val, y_val)
        xgb_params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
        xgb_params.update(best_xgb_params)
        xgb_model = xgb.XGBRegressor(
            **xgb_params,
            early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"],
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # --- Random Forest ---
        rf_model = RandomForestRegressor(**RF_PARAMS)
        rf_model.fit(X_train, y_train)

        # --- Ridge Regression (requires scaling + NaN imputation) ---
        train_means = X_train.mean().fillna(0)
        X_train_filled = X_train.fillna(train_means)
        X_val_filled = X_val.fillna(train_means)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_val_scaled = scaler.transform(X_val_filled)
        ridge_model = Ridge(**RIDGE_PARAMS)
        ridge_model.fit(X_train_scaled, y_train)

        # --- Determine weights from validation R² ---
        val_r2 = {
            "lgbm":  r2_score(y_val, lgbm_model.predict(X_val)),
            "xgb":   r2_score(y_val, xgb_model.predict(X_val)),
            "rf":    r2_score(y_val, rf_model.predict(X_val)),
            "ridge": r2_score(y_val, ridge_model.predict(X_val_scaled)),
        }
        logger.info(
            "Val R²  lgbm=%.4f  xgb=%.4f  rf=%.4f  ridge=%.4f",
            val_r2["lgbm"], val_r2["xgb"], val_r2["rf"], val_r2["ridge"],
        )

        r2_values = np.array(list(val_r2.values()))
        spread = r2_values.max() - r2_values.min()

        if spread < 0.02:
            weights = {name: 0.25 for name in val_r2}
            logger.info("Using equal weights (spread=%.3f < 0.02)", spread)
        else:
            clipped = np.clip(r2_values, 0, None)
            total = clipped.sum() or 1.0
            weights = dict(zip(val_r2.keys(), (clipped / total).tolist()))
            logger.info("Using weighted ensemble: %s", weights)

        models = {"lgbm": lgbm_model, "xgb": xgb_model, "rf": rf_model, "ridge": ridge_model}
        ensemble = EnsembleModel(models, weights, scaler=scaler, train_means=train_means)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as fh:
            pickle.dump({"model": ensemble, "feature_cols": available}, fh)
        logger.info("Ensemble model saved → %s", self.model_path)

        return ensemble

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

        # --- Per-model metrics ---
        print(f"\n{'Model':<20} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
        print("-" * 52)
        for name, sub_model in model.models.items():
            if name == "ridge" and model.scaler is not None:
                X_test_filled = pd.DataFrame(X_test).fillna(model.train_means)
                sub_pred = np.expm1(sub_model.predict(model.scaler.transform(X_test_filled)))
            else:
                sub_pred = sub_model.predict(X_test)
            print(f"  {name.upper():<18} {mean_absolute_error(y_test, sub_pred):>10.4f} "
                  f"{float(np.sqrt(mean_squared_error(y_test, sub_pred))):>10.4f} "
                  f"{r2_score(y_test, sub_pred):>8.4f}")
        print("-" * 52)

        # --- Ensemble ---
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)

        print(f"  {'ENSEMBLE':<18} {mae:>10.4f} {rmse:>10.4f} {r2:>8.4f}")
        print()
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

    def load_model(self) -> tuple[EnsembleModel, list[str]]:
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
