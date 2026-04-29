from fastapi import FastAPI
from .routes import root

app = FastAPI(title="fraud-detection-system", version="0.1.0")

app.include_router(root.router)


@app.get("/health", tags=["health"])
def health():
    """Endpoint de salud del sistema."""
    return {"status": "ok"}
