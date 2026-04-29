from fastapi import FastAPI
from .routes import root, auth_routes, dataset_routes, preprocessing_routes, model_routes, model_evaluation_routes, alert_routes

app = FastAPI(title="fraud-detection-system", version="0.1.0")

app.include_router(root.router)
app.include_router(auth_routes.router)
app.include_router(dataset_routes.router)
app.include_router(preprocessing_routes.router)
app.include_router(model_routes.router)
app.include_router(model_evaluation_routes.router)
app.include_router(alert_routes.router)


@app.get("/health", tags=["health"])
def health():
    """Endpoint de salud del sistema."""
    return {"status": "ok"}
