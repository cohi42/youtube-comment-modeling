"""Video -> key frames + 16 kHz mono audio.

Original: the extraction cells repeated three times in the Qwen-Omni notebook
(training asset preparation, video description inference, LoRA inference).
The three copies differed only in their numbers, so they are merged here into
one function; each caller passes the values it originally used.

Hybrid frame strategy
  - PySceneDetect ContentDetector(threshold=20) finds scene cuts.
  - Too few scenes  -> ffmpeg samples frames at a fixed interval instead.
  - Too many scenes -> scenes are sampled evenly down to `max_images`.
  - Scene frames are resized so the height is at most `target_height`.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass

import librosa
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images

AUDIO_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class ExtractionConfig:
    target_height: int
    min_images: int
    max_images: int
    ffmpeg_target_count: int = 12       # frames to aim for when falling back to ffmpeg
    cap_scenes: bool = True             # sample down to max_images when there are too many scenes
    skip_if_no_audio: bool = False      # stop (no frames) when audio extraction fails


# Values used in each part of the original notebook
TRAINING_ASSETS = ExtractionConfig(target_height=1024, min_images=5, max_images=15, skip_if_no_audio=True)
DESCRIPTION_INFERENCE = ExtractionConfig(target_height=448, min_images=4, max_images=8)
LORA_INFERENCE = ExtractionConfig(target_height=224, min_images=4, max_images=8, ffmpeg_target_count=8, cap_scenes=False)


def extract_audio(video_path: str, audio_path: str) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-y", audio_path],
            check=True,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False  # some Shorts have no audio track


def _resize_frames(frames_dir: str, target_height: int) -> None:
    for img_name in os.listdir(frames_dir):
        if not img_name.endswith(".jpg"):
            continue
        p = os.path.join(frames_dir, img_name)
        try:
            with Image.open(p) as img:
                if img.height > target_height:
                    ratio = target_height / img.height
                    new_w = int(img.width * ratio)
                    img.resize((new_w, target_height), Image.Resampling.LANCZOS).save(p)
        except Exception:
            pass


def extract_frames(video_path: str, frames_dir: str, cfg: ExtractionConfig) -> str:
    """Write frame_*.jpg into frames_dir and return the method used."""
    video = open_video(video_path)
    try:
        duration_sec = len(video) / video.frame_rate
    except Exception:
        duration_sec = 0

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=20.0))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    if len(scene_list) < cfg.min_images:
        interval = max(2.0, duration_sec / cfg.ffmpeg_target_count) if duration_sec > 0 else 2.0
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vf", f"fps=1/{interval},scale=-2:{cfg.target_height}",
                "-q:v", "2", "-y",
                os.path.join(frames_dir, "frame_%03d.jpg"),
            ],
            stderr=subprocess.DEVNULL,
        )
        return "ffmpeg"

    if cfg.cap_scenes and len(scene_list) > cfg.max_images:
        step = len(scene_list) / cfg.max_images
        scene_list = [scene_list[int(i * step)] for i in range(cfg.max_images)]

    save_images(
        scene_list, video, num_images=1, image_extension="jpg", encoder_param=95,
        output_dir=frames_dir, image_name_template="frame_$SCENE_NUMBER",
    )
    _resize_frames(frames_dir, cfg.target_height)
    return "scenedetect"


def extract_media(video_path: str, out_dir: str, cfg: ExtractionConfig) -> tuple[str, str]:
    """Create out_dir/frames/*.jpg and out_dir/audio.wav. Returns (frames_dir, audio_path)."""
    frames_dir = os.path.join(out_dir, "frames")
    audio_path = os.path.join(out_dir, "audio.wav")
    os.makedirs(frames_dir, exist_ok=True)

    audio_ok = extract_audio(video_path, audio_path)
    if not audio_ok and cfg.skip_if_no_audio:
        return frames_dir, audio_path

    try:
        extract_frames(video_path, frames_dir, cfg)
    except Exception as e:
        # LoRA-inference notebook fallback: first 5 frames at 1 fps
        print(f"  frame extraction error (ffmpeg fallback): {e}")
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vf", f"fps=1,scale=-2:{cfg.target_height}",
                "-vframes", "5", "-q:v", "2", "-y",
                os.path.join(frames_dir, "frame_%03d.jpg"),
            ],
            stderr=subprocess.DEVNULL,
        )
    return frames_dir, audio_path


def reset_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def load_media_for_qwen(
    frames_dir: str,
    audio_path: str,
    max_frames: int | None = None,
    audio_duration: float | None = None,
) -> tuple[list, list, list[dict]]:
    """Load frames/audio and build the matching Qwen chat content placeholders.

    Returns (images, audios, content) where content is
    [{"type": "image"}, ..., {"type": "audio"}].
    """
    images, audios, content = [], [], []

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if max_frames is not None:
        frame_files = frame_files[:max_frames]
    for f in frame_files:
        try:
            images.append(Image.open(f).convert("RGB"))
            content.append({"type": "image"})
        except Exception:
            pass

    if os.path.exists(audio_path):
        try:
            y, _ = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, duration=audio_duration)
            audios.append(y)
            content.append({"type": "audio"})
        except Exception:
            pass

    return images, audios, content
