"""
FastAPI GPU backend for the Independent AI Generation Hub.
Serial job queue ensures exactly one CUDA workload runs at a time (OOM-safe).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from independent_generator import IndependentAIHub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("gpu_server")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

JobType = Literal["text-to-image", "text-to-video", "image-to-video"]

tasks: dict[str, dict[str, Any]] = {}
job_queue: asyncio.Queue[str] = asyncio.Queue()
hub: IndependentAIHub | None = None
_worker_task: asyncio.Task[None] | None = None

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}


class InternalStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskResponse(BaseModel):
    task_id: str
    status: str


class StatusResponse(BaseModel):
    task_id: str
    status: str  # processing | completed | failed
    job_type: str | None = None
    prompt: str | None = None
    file_url: str | None = None
    error: str | None = None


class PromptBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)


def _public_status(internal: InternalStatus) -> str:
    if internal in (InternalStatus.QUEUED, InternalStatus.PROCESSING):
        return "processing"
    if internal == InternalStatus.COMPLETED:
        return "completed"
    return "failed"


def _enqueue(
    job_type: JobType,
    prompt: str,
    input_image: Path | None = None,
    task_id: str | None = None,
) -> str:
    task_id = task_id or str(uuid.uuid4())
    ext = ".jpg" if job_type == "text-to-image" else ".mp4"
    tasks[task_id] = {
        "task_id": task_id,
        "job_type": job_type,
        "prompt": prompt,
        "input_image": str(input_image) if input_image else None,
        "output_file": STATIC_DIR / f"{task_id}{ext}",
        "status": InternalStatus.QUEUED,
        "file_url": None,
        "error": None,
    }
    job_queue.put_nowait(task_id)
    logger.info("Queued %s task %s (queue size: %s)", job_type, task_id, job_queue.qsize())
    return task_id


def _run_task(task_id: str) -> None:
    record = tasks[task_id]
    job_type: JobType = record["job_type"]
    prompt: str = record["prompt"]
    output_file: Path = record["output_file"]

    try:
        assert hub is not None
        if job_type == "text-to-image":
            hub.generate_text_to_image(prompt, output_file)
            record["file_url"] = f"/static/{output_file.name}"
        elif job_type == "text-to-video":
            hub.generate_text_to_video(prompt, output_file)
            record["file_url"] = f"/static/{output_file.name}"
        elif job_type == "image-to-video":
            image_path = record.get("input_image")
            if not image_path:
                raise ValueError("Missing input image for image-to-video task.")
            hub.generate_image_to_video(image_path, prompt, output_file)
            record["file_url"] = f"/static/{output_file.name}"
        else:
            raise ValueError(f"Unknown job type: {job_type}")

        record["status"] = InternalStatus.COMPLETED
        record["error"] = None
        logger.info("Task %s completed (%s).", task_id, job_type)
    except Exception as exc:  # noqa: BLE001
        record["status"] = InternalStatus.FAILED
        record["error"] = str(exc)
        logger.exception("Task %s failed (%s).", task_id, job_type)


async def _queue_worker() -> None:
    """Strict serial GPU worker — one generation at a time."""
    loop = asyncio.get_running_loop()
    while True:
        task_id = await job_queue.get()
        record = tasks.get(task_id)
        if record is None:
            job_queue.task_done()
            continue

        record["status"] = InternalStatus.PROCESSING
        logger.info("Worker processing task %s (%s)", task_id, record.get("job_type"))
        try:
            await loop.run_in_executor(None, _run_task, task_id)
        finally:
            job_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global hub, _worker_task

    logger.info("Booting Independent AI Hub (model download may take a while)...")
    hub = IndependentAIHub()
    _worker_task = asyncio.create_task(_queue_worker())
    logger.info("GPU server online — listening for jobs.")
    yield

    if _worker_task is not None:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    logger.info("GPU server stopped.")


app = FastAPI(
    title="Independent AI Generation Hub",
    description="Local GPU API for text-to-image, text-to-video, and image-to-video.",
    version="2.0.0",
    lifespan=lifespan,
)

# allow_credentials must be False when allow_origins is "*" (browser CORS rule).
# Vercel frontends are covered via allow_origin_regex; ngrok tunnel is same-origin to API from browser POV.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate/text-to-image", response_model=TaskResponse)
async def generate_text_to_image(body: PromptBody) -> TaskResponse:
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")
    task_id = _enqueue("text-to-image", prompt)
    return TaskResponse(task_id=task_id, status="processing")


@app.post("/api/generate/text-to-video", response_model=TaskResponse)
async def generate_text_to_video(body: PromptBody) -> TaskResponse:
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")
    task_id = _enqueue("text-to-video", prompt)
    return TaskResponse(task_id=task_id, status="processing")


@app.post("/api/generate/image-to-video", response_model=TaskResponse)
async def generate_image_to_video(
    prompt: str = Form(..., min_length=1, max_length=4000),
    image: UploadFile = File(...),
) -> TaskResponse:
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    content_type = (image.content_type or "").lower()
    if content_type and content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{content_type}'. Use JPEG, PNG, or WebP.",
        )

    suffix = Path(image.filename or "upload.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    task_id = str(uuid.uuid4())
    upload_path = UPLOADS_DIR / f"{task_id}{suffix}"

    try:
        with upload_path.open("wb") as handle:
            shutil.copyfileobj(image.file, handle)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to store upload: {exc}") from exc

    _enqueue("image-to-video", prompt, input_image=upload_path, task_id=task_id)
    return TaskResponse(task_id=task_id, status="processing")


@app.get("/api/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str) -> StatusResponse:
    record = tasks.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    internal: InternalStatus = record["status"]
    return StatusResponse(
        task_id=task_id,
        status=_public_status(internal),
        job_type=record.get("job_type"),
        prompt=record.get("prompt"),
        file_url=record.get("file_url"),
        error=record.get("error"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gpu_server:app", host="0.0.0.0", port=8000, reload=False)
