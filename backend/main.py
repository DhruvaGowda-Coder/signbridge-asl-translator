import os
import logging
import asyncio
import math
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("signbridge")

# --- Thread pool for blocking ML inference ---
_executor = ThreadPoolExecutor(max_workers=4)

# --- Request hardening ---
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", "250000"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "3000"))
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "5"))
_request_buckets: dict[str, deque[float]] = defaultdict(deque)

# --- Predictors (loaded once at startup) ---
static_predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once on startup; clean up on shutdown."""
    global static_predictor
    from predictor import StaticPredictor

    logger.info("Loading ML models...")
    static_predictor = StaticPredictor()

    if static_predictor.model is None:
        logger.warning("Static model not loaded \u2013 predictions will return 'Unknown'.")

    yield  # app is running

    _executor.shutdown(wait=False)
    logger.info("Shutdown complete.")

app = FastAPI(title="SignBridge API", lifespan=lifespan)

# --- CORS (explicit origins, no wildcard+credentials conflict) ---
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

# CORS middleware will be added after custom middlewares to ensure it's outermost

@app.middleware("http")
async def limit_prediction_request_size(request: Request, call_next):
    if request.url.path.startswith("/predict/"):
        content_length = request.headers.get("content-length")
        try:
            request_size = int(content_length) if content_length else 0
        except ValueError:
            request_size = MAX_REQUEST_BYTES + 1

        if request_size > MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large."},
            )
    return await call_next(request)

@app.middleware("http")
async def rate_limit_prediction_requests(request: Request, call_next):
    if request.url.path.startswith("/predict/"):
        client_host = request.client.host if request.client else "unknown"
        now = time.monotonic()
        bucket = _request_buckets[client_host]

        while bucket and bucket[0] <= now - RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()

        if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many prediction requests. Try again later."},
            )

        bucket.append(now)

    return await call_next(request)

# --- CORS (explicit origins, no wildcard+credentials conflict) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request / Response models ---
class StaticRequest(BaseModel):
    landmarks: list[float]

    @field_validator("landmarks")
    @classmethod
    def landmarks_must_be_finite(cls, landmarks: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in landmarks):
            raise ValueError("Landmarks must contain only finite floats.")
        return landmarks



# --- Routes ---
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "SignBridge"}

@app.post("/predict/static")
async def predict_static(req: StaticRequest):
    if len(req.landmarks) != 63:
        raise HTTPException(status_code=400, detail="Landmarks must contain exactly 63 floats.")

    if static_predictor is None:
        raise HTTPException(status_code=503, detail="Static predictor is not initialized.")

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, static_predictor.predict, req.landmarks),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("Static prediction timed out after %.2f seconds.", INFERENCE_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="Static prediction timed out.")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
