"""
Training Manager for NextItNet Model
Handles model training, validation, and checkpointing
"""

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Any

from collections.abc import Callable
from datetime import datetime
import threading
import time

from local_config import MODEL_CONFIG, DATA_DIR, MODELS_DIR


class SeqDataset(Dataset):
    """Dataset for sequential recommendations."""

    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data["input_seqs"])

    def __getitem__(self, idx):
        return {
            "input_seq": torch.LongTensor(self.data["input_seqs"][idx]),
            "target": torch.LongTensor([self.data["target_items"][idx]])[0],
        }


class TrainingJob:
    """Represents a training job with progress tracking."""

    def __init__(self, job_id: str, config: dict[str, Any]):
        self.job_id = job_id
        self.config = config
        self.status = "pending"  # pending, running, completed, failed, cancelled
        self.progress = 0.0  # 0-100
        self.current_epoch = 0
        self.total_epochs = config.get("epochs", 30)
        self.metrics = {}
        self.best_metrics = {}
        self.error_message = None
        self.started_at = None
        self.completed_at = None
        self.model_path = None

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for API responses."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "metrics": self.metrics,
            "best_metrics": self.best_metrics,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "model_path": self.model_path,
            "config": self.config,
        }


class TrainingManager:
    """Manages training jobs and model training."""

    def __init__(self):
        self.jobs: dict[str, TrainingJob] = {}
        self.active_job: str | None = None
        self._lock = threading.Lock()

    def create_job(self, config: dict[str, Any]) -> str:
        """Create a new training job."""
        job_id = (
            f"train_{int(time.time())}_{config.get('version', 'v1').replace('.', '_')}"
        )

        with self._lock:
            job = TrainingJob(job_id, config)
            self.jobs[job_id] = job

        return job_id

    def get_job(self, job_id: str) -> TrainingJob | None:
        """Get job by ID."""
        return self.jobs.get(job_id)

    def start_training(self, job_id: str, progress_callback: Callable | None = None):
        """Start training in a background thread."""
        job = self.get_job(job_id)
        if not job:
            return False

        if self.active_job is not None:
            job.status = "failed"
            job.error_message = "Another training job is already running"
            return False

        self.active_job = job_id
        thread = threading.Thread(
            target=self._train_model, args=(job, progress_callback)
        )
        thread.daemon = True
        thread.start()

        return True

    def _train_model(self, job: TrainingJob, progress_callback: Callable | None = None):
        """Train the model (runs in background thread)."""
        try:
            job.status = "running"
            job.started_at = datetime.utcnow().isoformat()

            # Import model here to avoid circular imports
            from recommender_nextitnet import NextItNet

            # Setup
            device = torch.device(
                "cuda"
                if job.config.get("use_gpu", True) and torch.cuda.is_available()
                else "cpu"
            )

            # Load data
            train_dataset = SeqDataset(DATA_DIR / "train.pkl")
            val_dataset = SeqDataset(DATA_DIR / "val.pkl")

            batch_size = job.config.get("batch_size", 256)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Create model
            model = NextItNet(
                num_items=MODEL_CONFIG["num_items"],
                embedding_dim=MODEL_CONFIG["embedding_dim"],
                num_blocks=MODEL_CONFIG["num_blocks"],
                kernel_size=MODEL_CONFIG["kernel_size"],
                dilation_rates=MODEL_CONFIG["dilations"],
                dropout=MODEL_CONFIG["dropout"],
            ).to(device)

            # Optimizer
            optimizer = optim.Adam(
                model.parameters(), lr=job.config.get("learning_rate", 0.001)
            )

            # Training loop
            best_val_loss = float("inf")
            patience = 5 if job.config.get("early_stopping", True) else job.total_epochs
            patience_counter = 0

            for epoch in range(1, job.total_epochs + 1):
                # Check if cancelled
                if job.status == "cancelled":
                    break

                job.current_epoch = epoch
                job.progress = (epoch / job.total_epochs) * 100

                # Train
                model.train()
                train_loss = 0.0
                for batch in train_loader:
                    input_seq = batch["input_seq"].to(device)
                    target = batch["target"].to(device)

                    optimizer.zero_grad()
                    logits = model(input_seq)

                    # Negative sampling loss
                    loss = nn.CrossEntropyLoss()(logits[:, -1, :], target)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        input_seq = batch["input_seq"].to(device)
                        target = batch["target"].to(device)
                        logits = model(input_seq)
                        loss = nn.CrossEntropyLoss()(logits[:, -1, :], target)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Update metrics
                job.metrics = {
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "epoch": epoch,
                }

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    job.best_metrics = job.metrics.copy()
                    job.best_metrics["best_val_loss"] = float(best_val_loss)

                    # Save checkpoint
                    version = job.config.get("version", "v1")
                    model_filename = f"nextitnet_{version}_{job.job_id}.pth"
                    job.model_path = str(MODELS_DIR / model_filename)

                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "config": MODEL_CONFIG,
                        },
                        job.model_path,
                    )

                    patience_counter = 0
                else:
                    patience_counter += 1

                # Callback for progress updates
                if progress_callback:
                    progress_callback(job)

                # Early stopping
                if (
                    job.config.get("early_stopping", True)
                    and patience_counter >= patience
                ):
                    job.metrics["early_stopped"] = True
                    job.metrics["stopped_at_epoch"] = epoch
                    break

            job.status = "completed"
            job.progress = 100.0
            job.completed_at = datetime.utcnow().isoformat()

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow().isoformat()

        finally:
            self.active_job = None
            if progress_callback:
                progress_callback(job)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job."""
        job = self.get_job(job_id)
        if not job or job.status != "running":
            return False

        job.status = "cancelled"
        return True

    def list_jobs(self) -> list:
        """List all training jobs."""
        return [job.to_dict() for job in self.jobs.values()]


# Global training manager instance
training_manager = TrainingManager()
