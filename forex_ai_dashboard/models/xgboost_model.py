from __future__ import annotations
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from ..utils.logger import logger


class XGBoostModel:
    """
    Thin wrapper providing a stable interface + persistence.
    """

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        n_jobs: int = -1,
        random_state: int = 42,
        early_stopping_rounds: int = 50,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.model = XGBRegressor(**self.params)
        logger.info(f"Initialized XGBoostModel: {self.params}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs.update(
                dict(eval_set=[(X_val, y_val)], early_stopping_rounds=self.early_stopping_rounds, verbose=False)
            )
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    # for rolling_validation compatibility
    def fit(self, X, y):
        self.model.fit(X, y, verbose=False)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate_rmse(self, X, y) -> float:
        preds = self.predict(X)
        return mean_squared_error(y, preds, squared=False)

    def save(self, path: str):
        self.model.save_model(path)
        logger.info(f"Saved XGBoost model to {path}")

    def load(self, path: str):
        self.model = xgb.XGBRegressor(**self.params)
        self.model.load_model(path)
        logger.info(f"Loaded XGBoost model from {path}")
        return self
