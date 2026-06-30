from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class JobStore:
    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, dict[str, Any]] = {}

    def create_entry(self, job_id: str) -> None:
        self._jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "result": None,
            "error": None,
        }

    def submit(self, job_id: str, fn: Callable[[], Any]) -> None:
        self._executor.submit(self._run, job_id, fn)

    def _run(self, job_id: str, fn: Callable[[], Any]) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        try:
            job["status"] = "running"
            result = fn()
            job["status"] = "done"
            job["result"] = result
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    def new_job_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def prune(self, keep: int = 100) -> None:
        keys = sorted(self._jobs.keys(), reverse=True)
        for k in keys[keep:]:
            del self._jobs[k]
