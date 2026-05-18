from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import root, auth_routes, dataset_routes, preprocessing_routes, model_routes, model_evaluation_routes, alert_routes, settings_routes
from .config import settings


app = FastAPI(title="fraud-detection-system", version="0.1.0")

# Configure CORS for frontend dev (Vite) and any allowed origins from env
default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
env_orig = []
try:
    env_val = getattr(settings, "cors_origins", None)
    if env_val:
        # allow comma-separated list
        env_orig = [o.strip() for o in str(env_val).split(",") if o.strip()]
except Exception:
    env_orig = []

allowed_origins = list(dict.fromkeys(default_origins + env_orig))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(root.router)
app.include_router(auth_routes.router)
app.include_router(dataset_routes.router)
app.include_router(preprocessing_routes.router)
app.include_router(model_routes.router)
app.include_router(model_evaluation_routes.router)
app.include_router(alert_routes.router)
app.include_router(settings_routes.router)


@app.get("/health", tags=["health"])
def health():
    """Endpoint de salud del sistema."""
    return {"status": "ok"}
