from __future__ import annotations
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from ..utils.logger import logger


class CatBoostModel:
    def __init__(
        self,
        iterations: int = 800,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        loss_function: str = "RMSE",
        random_seed: int = 42,
        early_stopping_rounds: int = 50,
    ):
        self.params = dict(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            loss_function=loss_function,
            eval_metric="RMSE",
            random_seed=random_seed,
            verbose=False,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.model = CatBoostRegressor(**self.params)
        logger.info(f"Initialized CatBoostModel: {self.params}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs.update(dict(eval_set=(X_val, y_val), early_stopping_rounds=self.early_stopping_rounds))
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    # for rolling_validation compatibility
    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate_rmse(self, X, y) -> float:
        preds = self.predict(X)
        return mean_squared_error(y, preds, squared=False)

    def save(self, path: str):
        self.model.save_model(path)
        logger.info(f"Saved CatBoost model to {path}")

    def load(self, path: str):
        self.model = CatBoostRegressor(**self.params)
        self.model.load_model(path)
        logger.info(f"Loaded CatBoost model from {path}")
        return self
