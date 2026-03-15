"""Training API routes."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from yolo_demo.training import TrainingConfig, Trainer, TrainingResult
from yolo_demo.api.schemas import TrainingRequest, TrainingStatusResponse

router = APIRouter(prefix="/train", tags=["training"])

# In-memory job storage (use Redis in production)
_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)


def _run_training_job(job_id: str, config: TrainingConfig) -> None:
    """Run training job in background."""
    try:
        trainer = Trainer(config)
        result = trainer.train()

        _jobs[job_id]["status"] = "completed" if result.success else "failed"
        _jobs[job_id]["progress"] = 1.0
        _jobs[job_id]["result"] = result

        if result.success:
            _jobs[job_id]["model_path"] = result.model_path
            _jobs[job_id]["metrics"] = result.metrics
        else:
            _jobs[job_id]["error"] = result.error

    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


@router.post("", response_model=dict[str, str])
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start a training job.

    Returns a job_id that can be used to check status.
    """
    job_id = str(uuid.uuid4())

    config = TrainingConfig(
        model=request.model or "yolov8n.pt",
        data=request.data_yaml,
        epochs=request.epochs,
        batch=request.batch_size,
        imgsz=request.imgsz,
        lr0=request.lr0,
    )

    # Initialize job
    _jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "config": config,
        "error": None,
        "model_path": None,
        "metrics": {},
    }

    # Start training in background
    background_tasks.add_task(_run_training_job, job_id, config)

    return {"job_id": job_id, "status": "pending"}


@router.get("/{job_id}/status", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    """Get the status of a training job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    return TrainingStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        metrics=job.get("metrics", {}),
        error=job.get("error"),
        model_path=job.get("model_path"),
    )


@router.delete("/{job_id}")
async def cancel_training(job_id: str):
    """Cancel a training job (not fully implemented)."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # Note: This is a simple implementation
    # In production, you'd need to properly stop the training process
    if _jobs[job_id]["status"] in ["pending", "running"]:
        _jobs[job_id]["status"] = "cancelled"
        return {"status": "cancelled"}

    return {"status": "cannot_cancel", "message": "Job already completed or failed"}
