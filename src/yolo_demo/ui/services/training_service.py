"""Training service — decouples Gradio callbacks from training lifecycle.

Provides TrainingSession (manages one training run) and TrainingSessionManager
(multi-session queue for a single Gradio user session).
"""

import logging
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from ...training.trainer import Trainer, TrainingConfig

logger = logging.getLogger(__name__)


class _SessionLogHandler(logging.Handler):
    """Log handler that forwards log records into a queue.Queue.

    Used to stream training logs to the Gradio UI textbox.
    Uses the 'ultralytics' logger (NOT root) to isolate sessions.
    """

    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put_nowait(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


@dataclass
class TrainingJobConfig:
    """Simplified training configuration for the WebUI."""

    model_path: str
    dataset_path: str
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    lr0: float = 0.01
    output_dir: Optional[str] = None

    def to_training_config(self) -> TrainingConfig:
        """Convert to the full TrainingConfig used by the trainer."""
        cfg = TrainingConfig(
            model=self.model_path,
            epochs=int(self.epochs),
            batch=int(self.batch_size),
            imgsz=int(self.imgsz),
            lr0=float(self.lr0),
        )
        if self.output_dir:
            cfg.project = self.output_dir
        return cfg


class TrainingSession:
    """Manages a single training run lifecycle.

    Lifecycle:
        pending → (start) → running → (done | error | cancel) → finished

    The session owns a daemon thread, a message queue, and a log handler.
    Callers use poll() to receive log lines and completion signals.
    """

    def __init__(self, job_id: str, config: TrainingJobConfig) -> None:
        self.job_id = job_id
        self.config = config
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status: str = "pending"
        self._model_path: Optional[str] = None
        self._error: Optional[str] = None
        self._msg_queue: queue.Queue = queue.Queue()
        self._log_handler: Optional[_SessionLogHandler] = None

    @property
    def status(self) -> str:
        return self._status

    @property
    def model_path(self) -> Optional[str]:
        return self._model_path

    @property
    def error(self) -> Optional[str]:
        return self._error

    def start(self) -> None:
        """Start training in a background daemon thread.

        If already running, this is a no-op.
        """
        if self._status == "running":
            return

        self._status = "running"

        # Attach a per-session log handler to the ultralytics logger (NOT root).
        self._log_handler = _SessionLogHandler(self._msg_queue)
        self._log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logging.getLogger("ultralytics").addHandler(self._log_handler)

        self._thread = threading.Thread(target=self._training_thread_fn, daemon=True)
        self._thread.start()

    def _training_thread_fn(self) -> None:
        """Internal function executed in the training thread."""
        log_handler = self._log_handler  # capture for finally
        try:
            cfg = self.config.to_training_config()
            trainer = Trainer(cfg)
            trainer.load_pretrained()
            yolo_model = trainer.model
            assert yolo_model is not None, "Failed to load model"

            stop_ev = self._stop_event

            def _on_epoch_end(t):
                if stop_ev.is_set():
                    t.epoch = t.epochs
                    logger.info(
                        "Stop requested — finishing current epoch and exiting."
                    )

            yolo_model.add_callback("on_train_epoch_end", _on_epoch_end)

            logger.info("Starting training with config: %s", cfg.to_dict())
            result = trainer.train(data_yaml=self.config.dataset_path)

            if result.success:
                self._model_path = result.model_path or ""
                self._msg_queue.put(("done", self._model_path))
                self._status = "completed"
            else:
                err = result.error or "Unknown training error"
                self._error = err
                self._msg_queue.put(("error", err))
                self._status = "error"

        except Exception as e:
            err = str(e)
            self._error = err
            self._msg_queue.put(("error", err))
            self._status = "error"
        finally:
            if log_handler is not None:
                try:
                    logging.getLogger("ultralytics").removeHandler(log_handler)
                except Exception:
                    pass

    def cancel(self) -> None:
        """Request cancellation at next epoch boundary."""
        self._stop_event.set()
        if self._status == "running":
            self._status = "cancelled"

    def poll(self, timeout: float = 0.5) -> Optional[Union[str, Tuple[str, str]]]:
        """Poll the message queue for log lines or completion signals.

        Returns:
            str: A log line to display in the UI.
            ("done", model_path): Training completed successfully.
            ("error", message): Training failed.
            None: No message was available within the timeout.
        """
        try:
            msg = self._msg_queue.get(timeout=timeout)
            return msg
        except queue.Empty:
            return None

    def is_active(self) -> bool:
        """Check if the training thread is still alive."""
        return self._thread is not None and self._thread.is_alive()

    def has_messages(self) -> bool:
        """Check if there are unread messages in the queue."""
        return not self._msg_queue.empty()

    def cleanup(self) -> None:
        """Release resources: remove log handler, join thread.

        Safe to call multiple times.
        """
        # Cancel first so the thread exits at next epoch boundary
        self.cancel()

        if self._log_handler is not None:
            try:
                logging.getLogger("ultralytics").removeHandler(self._log_handler)
            except Exception:
                pass
            self._log_handler = None

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None


class TrainingSessionManager:
    """Manages training sessions for a single Gradio session.

    Enforces max_concurrent limit (default 1) to prevent GPU OOM.
    Designed to be stored in gr.State() — one manager per browser session.
    """

    def __init__(self, max_concurrent: int = 1) -> None:
        self.max_concurrent = max_concurrent
        self._sessions: Dict[str, TrainingSession] = {}
        self._active_job_id: Optional[str] = None

    @property
    def active_count(self) -> int:
        return 1 if self._active_job_id is not None else 0

    @property
    def active_job_id(self) -> Optional[str]:
        return self._active_job_id

    def submit(self, config: TrainingJobConfig) -> str:
        """Submit a new training job.

        Starts immediately if no active job; otherwise returns the job_id
        but defers start until the active job completes.

        Returns:
            The job_id string.
        """
        job_id = str(uuid.uuid4())[:8]
        session = TrainingSession(job_id, config)
        self._sessions[job_id] = session

        if self._active_job_id is None:
            self._active_job_id = job_id
            session.start()

        return job_id

    def get(self, job_id: str) -> Optional[TrainingSession]:
        """Retrieve a session by job_id."""
        return self._sessions.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancel a training job. Returns True if the job was found."""
        session = self._sessions.get(job_id)
        if session is None:
            return False
        session.cancel()
        return True

    def on_job_finished(self, job_id: str) -> None:
        """Call when a job completes — marks it inactive and starts next."""
        if self._active_job_id == job_id:
            self._active_job_id = None

    def cleanup(self, job_id: str) -> None:
        """Clean up a specific session's resources."""
        session = self._sessions.pop(job_id, None)
        if session is not None:
            session.cleanup()
        if self._active_job_id == job_id:
            self._active_job_id = None

    def cleanup_all(self) -> None:
        """Clean up ALL sessions (e.g., on browser disconnect or shutdown)."""
        for job_id in list(self._sessions.keys()):
            self.cleanup(job_id)
        self._active_job_id = None
