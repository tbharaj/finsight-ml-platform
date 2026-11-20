from fastapi.testclient import TestClient

from finsight.api import create_app
from test_pipeline import trained_model


def test_api_predict_and_input_validation():
    model, _ = trained_model()
    with TestClient(create_app(model)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/predict", json={"text": "Profit is higher than expected"})
        assert response.status_code == 200
        assert set(response.json()) == {"label", "probabilities"}
        assert client.post("/predict", json={"text": " "}).status_code == 422
        assert client.post("/predict", json={"text": "x" * 2001}).status_code == 422
