from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time

REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total number of code review requests",
    ["endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "End-to-end request latency in seconds",
    ["endpoint"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]
)

TOKENS_GENERATED = Histogram(
    "llm_tokens_generated",
    "Number of tokens generated per request",
    buckets=[50, 100, 200, 300, 500, 750, 1024]
)

ACTIVE_REQUESTS = Gauge("llm_active_requests", "Requests currently being processed")
MODEL_INFO = Gauge("llm_model_info", "Loaded model info", ["model_name", "version"])

def record_model_info(model_name: str, version: str):
    MODEL_INFO.labels(model_name=model_name, version=version).set(1)

def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

class LatencyTimer:
    def __init__(self, histogram, labels):
        self.histogram = histogram
        self.labels = labels

    def __enter__(self):
        self.start = time.time()
        ACTIVE_REQUESTS.inc()
        return self

    def __exit__(self, *args):
        self.histogram.labels(**self.labels).observe(time.time() - self.start)
        ACTIVE_REQUESTS.dec()
