"""FraudPredictor: turns one raw PaySim transaction into a fraud decision.

Loads the artifacts saved by Phase 4/5 (model JSON + metadata JSON), rebuilds the
decision-time features exactly as in feature engineering, and applies the saved threshold.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger("fraud_predictor")

REQUIRED_FIELDS = ("step", "type", "amount", "oldbalanceOrg", "nameDest")
ALLOWED_TYPES = ("TRANSFER", "CASH_OUT")
NEAR_EMPTY_THRESHOLD = 0.95


class InvalidTransactionError(ValueError):
    """Raised when an incoming transaction is missing fields or has impossible values."""


class DestinationHistory:
    """Transactions received per destination account, kept per time step.

    Training defines dest_prior_txn_count as the number of transactions the destination received in
    strictly EARLIER steps. Keeping counts per step lets serving apply the identical rule.
    In production this state lives in Redis or a feature store; here it is an in-memory dict.
    """

    def __init__(self, counts: Mapping[str, Mapping[int, int]] | None = None) -> None:
        self._counts: dict[str, dict[int, int]] = {d: dict(steps) for d, steps in (counts or {}).items()}

    @classmethod
    def from_dataframe(cls, history: pd.DataFrame) -> "DestinationHistory":
        counts: dict[str, dict[int, int]] = {}
        for (dest, step), n in history.groupby(["nameDest", "step"]).size().items():
            counts.setdefault(dest, {})[int(step)] = int(n)
        return cls(counts)

    def prior_count(self, name_dest: str, step: int) -> int:
        return sum(n for s, n in self._counts.get(name_dest, {}).items() if s < step)

    def record(self, name_dest: str, step: int) -> None:
        steps = self._counts.setdefault(name_dest, {})
        steps[int(step)] = steps.get(int(step), 0) + 1

    def __len__(self) -> int:
        return len(self._counts)


@dataclass(frozen=True)
class Prediction:
    fraud_probability: float
    is_fraud: bool
    risk_band: str
    threshold: float
    feature_set: str
    model_name: str
    reasons: tuple = ()

    def to_dict(self) -> dict:
        return asdict(self)


class FraudPredictor:
    def __init__(self, model: xgb.XGBClassifier, meta: dict, history: DestinationHistory | None = None) -> None:
        self._model = model
        self._meta = meta
        self.features: list[str] = list(meta["features"])
        self.threshold: float = float(meta["threshold"])
        self.feature_set: str = meta["feature_set"]
        self.model_name: str = meta.get("model", "xgboost")
        self.history = history or DestinationHistory()

    @classmethod
    def from_artifacts(cls, model_dir: str | Path, tuned: bool = True,
                       history: DestinationHistory | None = None) -> "FraudPredictor":
        model_dir = Path(model_dir)
        suffix = "_tuned" if tuned and (model_dir / "model_meta_tuned.json").exists() else ""
        meta = json.loads((model_dir / f"model_meta{suffix}.json").read_text())
        model = xgb.XGBClassifier()
        model.load_model(str(model_dir / f"xgb_fraud{suffix}.json"))
        logger.info("loaded %s model with %d features, threshold %.4f", suffix or "baseline",
                    len(meta["features"]), meta["threshold"])
        return cls(model, meta, history)

    @staticmethod
    def validate(txn: Mapping) -> None:
        missing = [f for f in REQUIRED_FIELDS if f not in txn or txn[f] is None]
        if missing:
            raise InvalidTransactionError(f"missing fields: {missing}")
        if txn["type"] not in ALLOWED_TYPES:
            raise InvalidTransactionError(f"type must be one of {ALLOWED_TYPES}, got {txn['type']!r}")
        if not float(txn["amount"]) >= 0:
            raise InvalidTransactionError("amount must be zero or positive")
        if not float(txn["oldbalanceOrg"]) >= 0:
            raise InvalidTransactionError("oldbalanceOrg must be zero or positive")
        if int(txn["step"]) < 0:
            raise InvalidTransactionError("step cannot be negative")

    def build_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        amount = raw["amount"].astype("float64")
        old_balance = raw["oldbalanceOrg"].astype("float64")
        hour = (raw["step"].astype("int64") % 24).astype("int8")
        out = pd.DataFrame(index=raw.index)
        out["log_amount"] = np.log1p(amount)
        out["is_transfer"] = (raw["type"] == "TRANSFER").astype("int8")
        out["hour"] = hour
        out["is_night"] = hour.between(0, 5).astype("int8")
        out["dest_prior_txn_count"] = pd.Series(
            [self.history.prior_count(d, int(st)) for d, st in zip(raw["nameDest"], raw["step"])],
            index=raw.index).astype("int32")
        out["dest_is_merchant"] = raw["nameDest"].str.startswith("M").astype("int8")
        out["amount_to_oldbalance_ratio"] = amount / (old_balance + 1)
        out["sender_near_empty"] = ((old_balance > 0) & (amount >= NEAR_EMPTY_THRESHOLD * old_balance)).astype("int8")
        out["sender_zero_balance"] = (old_balance == 0).astype("int8")
        return out[self.features]

    def predict_proba_batch(self, raw: pd.DataFrame) -> np.ndarray:
        for record in raw.to_dict("records"):
            self.validate(record)
        return self._model.predict_proba(self.build_features(raw))[:, 1]

    def _band(self, probability: float) -> str:
        if probability >= self.threshold:
            return "HIGH"
        if probability >= self.threshold * 0.5:
            return "MEDIUM"
        return "LOW"

    def explain_batch(self, raw: pd.DataFrame, top_k: int = 3) -> list[tuple]:
        """Top-k SHAP contributions (log-odds) per row, largest absolute effect first."""
        features = self.build_features(raw)
        contributions = self._model.get_booster().predict(xgb.DMatrix(features), pred_contribs=True)[:, :-1]
        reasons = []
        for values, contrib in zip(features.to_numpy(), contributions):
            order = np.argsort(-np.abs(contrib))[:top_k]
            reasons.append(tuple((self.features[i], float(values[i]), float(contrib[i])) for i in order))
        return reasons

    def predict(self, txn: Mapping, record_history: bool = False, explain: bool = False,
                top_k: int = 3) -> Prediction:
        self.validate(txn)
        frame = pd.DataFrame([dict(txn)])
        probability = float(self.predict_proba_batch(frame)[0])
        reasons = self.explain_batch(frame, top_k)[0] if explain else ()
        if record_history:
            self.history.record(txn["nameDest"], int(txn["step"]))
        result = Prediction(
            fraud_probability=probability,
            is_fraud=probability >= self.threshold,
            risk_band=self._band(probability),
            threshold=self.threshold,
            feature_set=self.feature_set,
            model_name=self.model_name,
            reasons=reasons,
        )
        logger.info("scored %s -> %.4f (%s)", txn.get("nameDest"), probability, result.risk_band)
        return result

    def predict_batch(self, raw: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.predict_proba_batch(raw)
        return pd.DataFrame({"fraud_probability": probabilities,
                             "is_fraud": probabilities >= self.threshold}, index=raw.index)

    def __repr__(self) -> str:
        return (f"FraudPredictor(model={self.model_name!r}, feature_set={self.feature_set!r}, "
                f"n_features={len(self.features)}, threshold={self.threshold:.4f})")


def score_all(predictor: FraudPredictor, transactions: Iterable[Mapping]) -> list[Prediction]:
    return [predictor.predict(t) for t in transactions]
