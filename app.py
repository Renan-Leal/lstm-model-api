from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
import time
import psutil
import sys
import uuid

from tensorflow.keras.models import load_model
import joblib
from loguru import logger


# =========================================================
# LOG CONFIGURATION (Colored + Structured)
# =========================================================
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | "
           "<level>{level}</level> | "
           "req_id=<cyan>{extra[request_id]}</cyan> | "
           "<level>{message}</level>",
    level="INFO"
)


# =========================================================
# APP INITIALIZATION
# =========================================================
app = FastAPI(title="LSTM Stock Prediction API")

WINDOW_SIZE = 60

logger.info("Loading model and scaler...", request_id="SYSTEM")
model = load_model("modelo_lstm.keras")
scaler = joblib.load("scaler.pkl")
logger.success("Model loaded successfully", request_id="SYSTEM")


# =========================================================
# SCHEMAS
# =========================================================
class PriceInput(BaseModel):
    last_60_prices: list[float]


class PredictNDaysInput(BaseModel):
    last_60_prices: list[float]
    n_days: int


# =========================================================
# MIDDLEWARE - REQUEST TRACKING
# =========================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    bound_logger = logger.bind(request_id=request_id)

    bound_logger.info(
        "REQUEST START | method={} | path={} | ip={}",
        request.method,
        request.url.path,
        client_ip
    )

    try:
        response = await call_next(request)
        process_time = round(time.time() - start_time, 4)

        bound_logger.info(
            "REQUEST END | status={} | latency={}s",
            response.status_code,
            process_time
        )

        return response

    except Exception:
        process_time = round(time.time() - start_time, 4)
        bound_logger.exception(
            "REQUEST ERROR | latency={}s",
            process_time
        )
        raise


# =========================================================
# UTILITIES
# =========================================================
def get_resource_usage():
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent
    }


# =========================================================
# ENDPOINTS
# =========================================================
@app.get("/health")
def health(request: Request):
    log = logger.bind(request_id=request.state.request_id)
    resources = get_resource_usage()

    log.info(
        "HEALTH CHECK | cpu={}%% | memory={}%%",
        resources["cpu_percent"],
        resources["memory_percent"]
    )

    return {
        "status": "ok",
        "cpu_usage_percent": resources["cpu_percent"],
        "memory_usage_percent": resources["memory_percent"]
    }


@app.get("/model-info")
def model_info(request: Request):
    log = logger.bind(request_id=request.state.request_id)
    log.info("MODEL INFO REQUESTED")
    return {
        "model_type": "LSTM",
        "window_size": WINDOW_SIZE,
        "framework": "TensorFlow / Keras",
        "task": "Time series forecasting - closing price prediction"
    }


@app.post("/predict")
def predict_price(data: PriceInput, request: Request):
    log = logger.bind(request_id=request.state.request_id)

    if len(data.last_60_prices) != WINDOW_SIZE:
        log.warning("INVALID INPUT SIZE | expected=60")
        raise HTTPException(status_code=400, detail="Exactly 60 prices are required")

    start_time = time.time()

    try:
        prices = np.array(data.last_60_prices).reshape(-1, 1)
        scaled_prices = scaler.transform(prices)
        X_input = scaled_prices.reshape(1, WINDOW_SIZE, 1)

        scaled_prediction = model.predict(X_input, verbose=0)
        prediction = scaler.inverse_transform(scaled_prediction)

        latency = round(time.time() - start_time, 4)
        log.info("INFERENCE | type=single | latency={}s", latency)

        return {
            "predicted_close_price": float(prediction[0][0])
        }

    except Exception:
        log.exception("INFERENCE FAILED")
        raise HTTPException(status_code=500, detail="Inference failed")


@app.get("/explain")
def explain(request: Request):
    log = logger.bind(request_id=request.state.request_id)
    log.info("EXPLAIN REQUESTED")
    return {
        "explanation": (
            "The LSTM model uses the last 60 closing prices as input to learn "
            "temporal dependencies in the time series. Each prediction can be "
            "recursively fed back for multi-step forecasting."
        )
    }
