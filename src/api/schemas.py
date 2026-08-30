from pydantic import BaseModel


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    version: str = "1.1.0"


class Probabilities(BaseModel):
    cat: float
    dog: float


class PredictionResponse(BaseModel):
    label: str
    probabilities: Probabilities
    latency_ms: float
