"""Optional read-only inference API. No brokerage or execution integrations."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from finsight.inference import load_model, predict_text


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def create_app(model=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if model is not None:
            app.state.model = model
        else:
            path = os.getenv("FINSIGHT_MODEL_PATH", "artifacts/sentiment_model.joblib")
            app.state.model = load_model(path)
        yield

    app = FastAPI(title="FinSight Sentiment API", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict:
        try:
            return predict_text(app.state.model, request.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
