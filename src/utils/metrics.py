import time
from typing import Callable

from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class Endpoints:
    PREDICT: str = "/predict"
    PREDICT_BATCH: str = "/predict_batch"


PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Total number of predictions", ["endpoint"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

REQUESTS_TOTAL = Counter(
    "requests_total", "Total number of requests", ["method", "endpoint", "status"]
)

BATCH_SIZE_HISTOGRAM = Histogram(
    "batch_size", "Batch size distribution", buckets=(1, 2, 4, 8, 16, 32, 64, 128)
)

MODEL_INFO = Gauge("model_info", "Model version information", ["version"])

CACHE_HITS = Counter("cache_hits_total", "Total number of cache hits")
CACHE_MISSES = Counter("cache_misses_total", "Total number of cache misses")


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        response = await call_next(request)

        latency = time.time() - start_time
        endpoint = request.url.path
        method = request.method
        status = str(response.status_code)

        REQUEST_LATENCY.labels(method=method, endpoint=endpoint, status=status).observe(
            latency
        )
        REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status=status,
        ).inc()

        if endpoint == Endpoints.PREDICT:
            PREDICTIONS_TOTAL.labels(endpoint="single").inc()
            PREDICTION_LATENCY.labels(endpoint="single").observe(latency)
        elif endpoint == Endpoints.PREDICT_BATCH:
            PREDICTIONS_TOTAL.labels(endpoint="batch").inc()
            PREDICTION_LATENCY.labels(endpoint="batch").observe(latency)

        return response


def get_metrics() -> bytes:
    return generate_latest()
