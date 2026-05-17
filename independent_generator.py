"""
Independent AI Generation Hub — local engine for T2I, T2V, and I2V with synchronized audio.
All logs, comments, and user-facing errors are in English.
"""

from __future__ import annotations

import gc
import logging
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from audiocraft.data.audio import audio_write
from audiocraft.models import AudioGen
from diffusers import StableDiffusionPipeline, WanImageToVideoPipeline, WanPipeline
from diffusers.utils import export_to_video, load_image
from moviepy import AudioFileClip, VideoFileClip
from PIL import Image

logger = logging.getLogger(__name__)

# --- Model identifiers (with Diffusers-layout fallbacks where applicable) ---
IMAGE_MODEL_ID = "lykon/dreamshaper-8"
T2V_MODEL_IDS = ("Wan-AI/Wan2.1-T2V-1.3B", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
I2V_MODEL_IDS = (
    "Wan-AI/Wan2.1-I2V-1.3B",
    "Wan-AI/Wan2.1-I2V-1.3B-Diffusers",
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
)
AUDIO_MODEL_ID = "facebook/audiogen-medium"

VIDEO_FPS = 16
VIDEO_NUM_FRAMES = 81
AUDIO_DURATION_SEC = 5.0
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 768
IMAGE_STEPS = 35
IMAGE_GUIDANCE = 7.5

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, walking backwards"
)


class IndependentAIHub:
    """
    Production local hub loading DreamShaper (T2I), Wan T2V, Wan I2V, and AudioGen.
    VRAM is managed via float16 weights and Diffusers CPU offload + VAE slicing.
    """

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Inference will run on CPU and be very slow.")

        self._dtype = torch.float16
        logger.info("Initializing Independent AI Hub (loading all pipelines)...")

        self.image_pipe = self._load_image_pipeline()
        self.t2v_pipe = self._load_t2v_pipeline()
        self.i2v_pipe = self._load_i2v_pipeline()
        self.audio_model = self._load_audio_model()

        logger.info("Independent AI Hub is ready.")

    # ------------------------------------------------------------------ loaders
    def _apply_vram_optimizations(self, pipe) -> None:
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

    def _load_image_pipeline(self) -> StableDiffusionPipeline:
        logger.info("Loading text-to-image pipeline: %s", IMAGE_MODEL_ID)
        pipe = StableDiffusionPipeline.from_pretrained(
            IMAGE_MODEL_ID,
            torch_dtype=self._dtype,
            safety_checker=None,
        )
        self._apply_vram_optimizations(pipe)
        logger.info("Text-to-image pipeline ready.")
        return pipe

    def _load_t2v_pipeline(self) -> WanPipeline:
        last_error: Exception | None = None
        for model_id in T2V_MODEL_IDS:
            try:
                logger.info("Loading text-to-video pipeline: %s", model_id)
                pipe = WanPipeline.from_pretrained(model_id, torch_dtype=self._dtype)
                self._apply_vram_optimizations(pipe)
                logger.info("Text-to-video pipeline ready (%s).", model_id)
                return pipe
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Could not load T2V model %s: %s", model_id, exc)
        raise RuntimeError("Failed to load any Wan text-to-video checkpoint.") from last_error

    def _load_i2v_pipeline(self) -> WanImageToVideoPipeline:
        last_error: Exception | None = None
        for model_id in I2V_MODEL_IDS:
            try:
                logger.info("Loading image-to-video pipeline: %s", model_id)
                pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=self._dtype)
                self._apply_vram_optimizations(pipe)
                logger.info("Image-to-video pipeline ready (%s).", model_id)
                return pipe
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Could not load I2V model %s: %s", model_id, exc)
        raise RuntimeError("Failed to load any Wan image-to-video checkpoint.") from last_error

    def _load_audio_model(self) -> AudioGen:
        logger.info("Loading AudioGen: %s", AUDIO_MODEL_ID)
        model = AudioGen.get_pretrained(AUDIO_MODEL_ID)
        model.set_generation_params(duration=AUDIO_DURATION_SEC)
        logger.info("AudioGen ready.")
        return model

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _release_vram() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @contextmanager
    def _temp_workspace(self, prefix: str):
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            IndependentAIHub._release_vram()

    def _generate_silent_video_t2v(self, prompt: str, video_path: Path) -> None:
        logger.info("T2V: generating %s frames at %s fps.", VIDEO_NUM_FRAMES, VIDEO_FPS)
        with torch.inference_mode():
            result = self.t2v_pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_frames=VIDEO_NUM_FRAMES,
                height=480,
                width=832,
                guidance_scale=6.0,
                num_inference_steps=50,
            )
        export_to_video(result.frames[0], str(video_path), fps=VIDEO_FPS)

    def _prepare_i2v_image(self, image_path: Path) -> tuple[Image.Image, int, int]:
        """Resize input image to Wan I2V-friendly dimensions."""
        image = load_image(str(image_path))
        max_area = 480 * 832
        aspect_ratio = image.height / image.width
        mod_value = self.i2v_pipe.vae_scale_factor_spatial * self.i2v_pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        return image, height, width

    def _generate_silent_video_i2v(self, image_path: Path, prompt: str, video_path: Path) -> None:
        image, height, width = self._prepare_i2v_image(image_path)
        logger.info("I2V: generating %sx%s, %s frames.", width, height, VIDEO_NUM_FRAMES)
        with torch.inference_mode():
            result = self.i2v_pipe(
                image=image,
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=height,
                width=width,
                num_frames=VIDEO_NUM_FRAMES,
                guidance_scale=5.5,
                num_inference_steps=50,
            )
        export_to_video(result.frames[0], str(video_path), fps=VIDEO_FPS)

    def _generate_audio_wav(self, prompt: str, audio_path: Path) -> None:
        logger.info("Generating %.1fs of synchronized audio.", AUDIO_DURATION_SEC)
        wav = self.audio_model.generate([prompt])[0]
        audio_write(
            audio_path.with_suffix(""),
            wav.cpu(),
            self.audio_model.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )
        produced = audio_path.with_suffix(".wav")
        if produced != audio_path and produced.exists():
            produced.rename(audio_path)

    @staticmethod
    def _trim_clip(clip, end: float):
        if hasattr(clip, "subclipped"):
            return clip.subclipped(0, end)
        return clip.subclip(0, end)

    def _mux_video_audio(self, video_path: Path, audio_path: Path, output_path: Path) -> None:
        """Merge silent video and WAV into MP4; prefer copying the video stream."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            video_clip = VideoFileClip(str(video_path))
            audio_clip = AudioFileClip(str(audio_path))
            duration = min(video_clip.duration, audio_clip.duration)
            video_clip = self._trim_clip(video_clip, duration)
            audio_clip = self._trim_clip(audio_clip, duration)
            final = video_clip.with_audio(audio_clip)
            final.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                ffmpeg_params=["-c:v", "copy"],
                logger=None,
            )
            video_clip.close()
            audio_clip.close()
            final.close()
        except Exception as moviepy_exc:  # noqa: BLE001
            logger.warning("MoviePy mux failed (%s). Using ffmpeg fallback.", moviepy_exc)
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

    def _finalize_video_with_audio(self, prompt: str, silent_video: Path, destination: Path) -> Path:
        self._release_vram()
        with self._temp_workspace("aihub_audio_") as temp_dir:
            audio_wav = temp_dir / "audio.wav"
            self._generate_audio_wav(prompt, audio_wav)
            self._mux_video_audio(silent_video, audio_wav, destination)
        return destination.resolve()

    # ------------------------------------------------------------------ public API
    def generate_text_to_image(self, prompt: str, output_path: str | Path) -> Path:
        """Generate a high-quality JPEG from a text prompt."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt must not be empty.")

        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        logger.info("T2I: generating image for prompt (%.80s...)", prompt)
        with torch.inference_mode():
            result = self.image_pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                num_inference_steps=IMAGE_STEPS,
                guidance_scale=IMAGE_GUIDANCE,
            )

        image: Image.Image = result.images[0]
        image.save(destination, format="JPEG", quality=95, optimize=True, progressive=True)
        logger.info("T2I: saved %s", destination)
        self._release_vram()
        return destination

    def generate_text_to_video(self, prompt: str, output_path: str | Path) -> Path:
        """Generate video from text plus synchronized AudioGen audio; output MP4."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt must not be empty.")

        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        with self._temp_workspace("aihub_t2v_") as temp_dir:
            silent_video = temp_dir / "silent.mp4"
            self._generate_silent_video_t2v(prompt, silent_video)
            self._finalize_video_with_audio(prompt, silent_video, destination)

        logger.info("T2V: final MP4 saved to %s", destination)
        return destination

    def generate_image_to_video(
        self,
        image_path: str | Path,
        prompt: str,
        output_path: str | Path,
    ) -> Path:
        """Animate a source image with Wan I2V and mux synchronized audio into MP4."""
        prompt = prompt.strip()
        source = Path(image_path).resolve()
        if not prompt:
            raise ValueError("Prompt must not be empty.")
        if not source.is_file():
            raise FileNotFoundError(f"Input image not found: {source}")

        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        with self._temp_workspace("aihub_i2v_") as temp_dir:
            silent_video = temp_dir / "silent.mp4"
            self._generate_silent_video_i2v(source, prompt, silent_video)
            self._finalize_video_with_audio(prompt, silent_video, destination)

        logger.info("I2V: final MP4 saved to %s", destination)
        return destination
