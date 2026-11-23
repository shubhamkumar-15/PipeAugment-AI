import json
import logging
import os
import queue
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional
from uuid import UUID, uuid4

from fastapi import HTTPException
from app.core.config import Settings
from app.schemas.job import JobStatus, JobListResponse

logger = logging.getLogger(__name__)


class JobManager:
    """
    Thread-safe single-worker FIFO background job manager.
    Ensures at most ONE AI/CV/GPU job runs at any given time to avoid VRAM exhaustion.
    Persists job state JSON files under datasets/metadata/jobs/{job_id}.json.
    Supports runner injection for fast, deterministic unit testing without spawning GPU subprocesses.
    """

    def __init__(
        self,
        settings: Settings,
        runner_fn: Optional[Callable[[Dict[str, Any], Path], None]] = None,
    ):
        self.settings = settings
        self.jobs_dir: Path = settings.JOBS_DIR
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.runner_fn = runner_fn or self._default_runner

        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _try_warm_worker(self, job_spec: Dict[str, Any], spec_file: Path) -> bool:
        """Attempts to execute the job via the warm AI worker service (http://127.0.0.1:8001)."""
        import urllib.request
        import urllib.error

        warm_url = "http://127.0.0.1:8001"
        try:
            req = urllib.request.Request(f"{warm_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                if resp.status != 200:
                    return False
        except Exception:
            return False

        try:
            logger.info("Dispatching job %s to warm AI worker at %s/run_job", job_spec.get("job_id"), warm_url)
            payload = json.dumps({"spec_file": str(spec_file.resolve())}).encode("utf-8")
            post_req = urllib.request.Request(
                f"{warm_url}/run_job",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(post_req, timeout=300.0) as post_resp:
                if post_resp.status == 200:
                    logger.info("Warm worker completed job %s successfully.", job_spec.get("job_id"))
                    return True
                else:
                    logger.warning("Warm worker returned HTTP %d. Falling back to subprocess.", post_resp.status)
                    return False
        except Exception as exc:
            logger.warning("Warm worker execution failed (%s). Falling back to cold subprocess.", exc)
            return False

    def _default_runner(self, job_spec: Dict[str, Any], spec_file: Path) -> None:
        """First attempts execution via the persistent warm worker; falls back to cold subprocess."""
        if self._try_warm_worker(job_spec, spec_file):
            return
        logger.info("Warm worker unavailable or failed; executing cold subprocess.")
        self._default_subprocess_runner(job_spec, spec_file)

    def _default_subprocess_runner(self, job_spec: Dict[str, Any], spec_file: Path) -> None:
        """Executes the job using the isolated worker script in .venv-ai."""
        worker_script = self.settings.PROJECT_ROOT / "ai" / "scripts" / "run_api_job.py"
        ai_python = self.settings.AI_PYTHON_PATH

        if not worker_script.exists():
            raise FileNotFoundError(f"Worker script not found: {worker_script}")
        if not ai_python.exists():
            raise FileNotFoundError(f"AI python executable not found: {ai_python}")

        cmd = [str(ai_python), str(worker_script), "--job-spec", str(spec_file)]
        logger.info("Executing AI worker command: %s", " ".join(cmd))

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error("AI worker failed with code %d. Stderr: %s", res.returncode, res.stderr)
            raise RuntimeError(f"AI worker process failed (exit code {res.returncode}): {res.stderr.strip() or res.stdout.strip()}")

    def _save_job_status(self, job_id: UUID, status_data: Dict[str, Any]) -> None:
        """Atomically persist job status JSON file."""
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        target_file = self.jobs_dir / f"{job_id}.json"
        temp_file = self.jobs_dir / f".tmp_{job_id}.json"

        # Serialize datetime objects
        serialized = {}
        for k, v in status_data.items():
            if isinstance(v, (datetime,)):
                serialized[k] = v.isoformat()
            elif isinstance(v, UUID):
                serialized[k] = str(v)
            elif isinstance(v, list) and v and isinstance(v[0], UUID):
                serialized[k] = [str(item) for item in v]
            else:
                serialized[k] = v

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)
        os.replace(temp_file, target_file)

    def _load_job_status(self, job_id: UUID) -> Dict[str, Any]:
        """Loads job status JSON file from disk."""
        target_file = self.jobs_dir / f"{job_id}.json"
        if not target_file.is_file():
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Corrupted job file %s: %s", target_file, exc)
            raise HTTPException(status_code=500, detail="Corrupted job state file on disk.") from exc

    def _worker_loop(self) -> None:
        """Single background worker reading jobs sequentially from queue."""
        while True:
            try:
                job_spec = self._queue.get()
                if job_spec is None:
                    break

                job_id = UUID(str(job_spec["job_id"]))
                logger.info("Worker picked up job %s (type=%s)", job_id, job_spec["job_type"])

                spec_file = self.jobs_dir / f"spec_{job_id}.json"
                with open(spec_file, "w", encoding="utf-8") as f:
                    json.dump(job_spec, f, indent=2)

                try:
                    # Execute runner
                    self.runner_fn(job_spec, spec_file)
                except Exception as exc:
                    logger.error("Error executing job %s: %s", job_id, exc)
                    # Load current status and mark failed if not already updated
                    try:
                        curr = self._load_job_status(job_id)
                        curr["status"] = "failed"
                        curr["error_message"] = str(exc)
                        curr["finished_at"] = datetime.now(timezone.utc).isoformat()
                        self._save_job_status(job_id, curr)
                    except Exception as err:
                        logger.error("Failed to update failure status for job %s: %s", job_id, err)
                finally:
                    # Clean up temporary spec file if desired
                    if spec_file.exists():
                        try:
                            spec_file.unlink()
                        except OSError:
                            pass
                    self._queue.task_done()

            except Exception as loop_exc:
                logger.error("Unexpected worker loop exception: %s", loop_exc)

    def submit_job(
        self,
        job_type: str,
        parent_image_ids: List[UUID],
        parent_images: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> JobStatus:
        """Enqueues a new augmentation job and returns the initial queued JobStatus."""
        job_id = uuid4()
        now = datetime.now(timezone.utc)

        status_dict = {
            "job_id": str(job_id),
            "job_type": job_type,
            "status": "queued",
            "stage": "queued",
            "total_items": len(parent_image_ids),
            "completed_items": 0,
            "failed_items": 0,
            "parent_image_ids": [str(pid) for pid in parent_image_ids],
            "config": config,
            "created_at": now.isoformat(),
            "started_at": None,
            "finished_at": None,
            "generated_image_ids": [],
            "item_results": [],
            "error_message": None,
        }

        # Save initial queued state
        self._save_job_status(job_id, status_dict)

        job_spec = {
            "job_id": str(job_id),
            "job_type": job_type,
            "status_file": str(self.jobs_dir / f"{job_id}.json"),
            "parent_image_ids": [str(pid) for pid in parent_image_ids],
            "parent_images": parent_images,
            "config": config,
            "output_dirs": {
                "augmented_dir": str(self.settings.AUGMENTED_DIR),
                "metadata_dir": str(self.settings.DATASET_ROOT / "metadata"),
            },
        }

        # Put job into FIFO queue
        self._queue.put(job_spec)
        logger.info("Job %s enqueued successfully.", job_id)
        return JobStatus.model_validate(status_dict)

    def get_job(self, job_id: UUID) -> JobStatus:
        """Retrieves current job status by UUID."""
        raw = self._load_job_status(job_id)
        return JobStatus.model_validate(raw)

    def list_jobs(
        self,
        limit: int = 20,
        offset: int = 0,
        status_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
    ) -> JobListResponse:
        """Lists jobs sorted newest first with optional status and type filters."""
        if not self.jobs_dir.exists():
            return JobListResponse(items=[], total=0, limit=limit, offset=offset)

        all_jobs: List[JobStatus] = []
        for file_path in self.jobs_dir.glob("*.json"):
            if file_path.name.startswith((".", "spec_")):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                job_status = JobStatus.model_validate(data)

                # Filter by status if requested
                if status_filter and job_status.status != status_filter:
                    continue
                # Filter by type if requested
                if type_filter and job_status.job_type != type_filter:
                    continue

                all_jobs.append(job_status)
            except Exception as exc:
                logger.error("Skipping unreadable job file %s: %s", file_path.name, exc)

        # Sort newest first
        all_jobs.sort(key=lambda j: j.created_at, reverse=True)
        total = len(all_jobs)
        paged = all_jobs[offset : offset + limit]

        return JobListResponse(items=paged, total=total, limit=limit, offset=offset)

    def retry_job(
        self,
        job_id: UUID,
        parent_resolver: Callable[[UUID], Dict[str, Any]],
    ) -> JobStatus:
        """
        Re-queues failed items from a failed or partial_failed job.
        Creates and enqueues a new Job.
        """
        old_job = self.get_job(job_id)
        if old_job.status not in ("failed", "partial_failed"):
            raise HTTPException(
                status_code=400,
                detail=f"Only failed or partial_failed jobs can be retried (current status: '{old_job.status}').",
            )

        failed_parent_ids: List[UUID] = []
        if old_job.item_results:
            for item in old_job.item_results:
                if item.get("status") == "failed":
                    failed_parent_ids.append(UUID(item["parent_image_id"]))
        else:
            # If no item results were recorded, retry all parents
            failed_parent_ids = old_job.parent_image_ids

        if not failed_parent_ids:
            raise HTTPException(status_code=400, detail="No failed items found to retry in this job.")

        parent_images = [parent_resolver(pid) for pid in failed_parent_ids]

        # Re-submit with identical config
        return self.submit_job(
            job_type=old_job.job_type,
            parent_image_ids=failed_parent_ids,
            parent_images=parent_images,
            config=old_job.config,
        )


_job_manager_instance: Optional[JobManager] = None


def get_job_manager(settings: Settings) -> JobManager:
    """Singleton getter for FastAPI dependency injection."""
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = JobManager(settings=settings)
    return _job_manager_instance


def reset_job_manager_for_testing(new_instance: Optional[JobManager] = None) -> None:
    """Allows test fixtures to inject custom or mocked JobManager instances."""
    global _job_manager_instance
    _job_manager_instance = new_instance
