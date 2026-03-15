"""FastAPI application for YOLO Demo API."""

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import export, inference, training
from .schemas import HealthResponse, ModelsResponse

from ..inference import get_available_backend


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="YOLO Demo API",
        description="Lightweight real-time object detection system for edge computing",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(inference.router, prefix="/api/v1")
    app.include_router(training.router, prefix="/api/v1")
    app.include_router(export.router, prefix="/api/v1")

    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint."""
        return {
            "name": "YOLO Demo API",
            "version": "0.1.0",
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            backend=get_available_backend(),
            cuda_available=torch.cuda.is_available(),
            mps_available=torch.backends.mps.is_available(),
        )

    @app.get("/models", response_model=ModelsResponse, tags=["models"])
    async def list_models():
        """List available YOLO models."""
        # Default YOLOv8 models
        models = [
            {"name": "yolov8n", "path": "yolov8n.pt"},
            {"name": "yolov8s", "path": "yolov8s.pt"},
            {"name": "yolov8m", "path": "yolov8m.pt"},
            {"name": "yolov8l", "path": "yolov8l.pt"},
            {"name": "yolov8x", "path": "yolov8x.pt"},
        ]
        return ModelsResponse(
            models=[
                {"name": m["name"], "path": m["path"], "input_shape": [1, 3, 640, 640]}
                for m in models
            ]
        )

    return app


# Create the app instance
app = create_app()


def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Serve the API using uvicorn."""
    import uvicorn

    uvicorn.run("yolo_demo.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    serve()
