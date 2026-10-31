"""Request / response models for the fraud API."""
from typing import Literal

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    step: int = Field(..., ge=0, description="Hour of the simulation (1 step = 1 hour)", examples=[10])
    type: Literal["TRANSFER", "CASH_OUT"] = Field(..., description="Only these two types can be fraud in PaySim")
    amount: float = Field(..., ge=0, examples=[181.0])
    oldbalanceOrg: float = Field(..., ge=0, description="Sender balance before the transaction", examples=[181.0])
    nameDest: str = Field(..., min_length=2, description="Destination account id, e.g. C553264065 or M123", examples=["C553264065"])


class Reason(BaseModel):
    feature: str
    value: float
    contribution: float = Field(..., description="SHAP contribution in log-odds; positive pushes toward fraud")


class PredictionOut(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    threshold: float
    model_name: str
    feature_set: str
    reasons: list[Reason] = []


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)


class BatchItem(BaseModel):
    fraud_probability: float
    is_fraud: bool


class BatchResponse(BaseModel):
    count: int
    flagged: int
    results: list[BatchItem]


class ModelInfo(BaseModel):
    model_name: str
    feature_set: str
    features: list[str]
    threshold: float
    destinations_tracked: int
    validation_metrics: dict = {}
    test_metrics: dict = {}
