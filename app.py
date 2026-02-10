from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = FastAPI(title="API de Previsão de Ações - BBAS3")

model = load_model("modelo_lstm.keras")
scaler = joblib.load("scaler.pkl")

class PriceInput(BaseModel):
    last_60_prices: list[float]

@app.post("/predict")
def predict_price(data: PriceInput):

    if len(data.last_60_prices) != 60:
        return {"error": "É necessário fornecer exatamente 60 preços"}

    prices = np.array(data.last_60_prices).reshape(-1, 1)
    scaled_prices = scaler.transform(prices)
    X_input = scaled_prices.reshape(1, 60, 1)

    scaled_prediction = model.predict(X_input)
    prediction = scaler.inverse_transform(scaled_prediction)

    return {
        "predicted_close_price": float(prediction[0][0])
    }