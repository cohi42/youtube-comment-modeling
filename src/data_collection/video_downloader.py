"""Download the collected Shorts as mp4 files with yt-dlp.

Original: comment_collect/downloader.py (local script).

- Reads the nested collection JSON produced by youtube_shorts_scraper.py
- Android player client (avoids HTTP 403), <=720p, re-encoded to 10 fps
- Progress is saved to downloads/download_progress.json so a stopped run resumes
- 5 second pause between videos

Downloaded videos are not part of this repository.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROGRESS_FILE = "downloads/download_progress.json"
DOWNLOAD_FOLDER = "downloads/videos"


def load_videos(filename: str) -> dict:
    """Return {videoId: video} from any of the collection JSON layouts used in the project."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Invalid JSON: {filename}")
        sys.exit(1)

    # channels[].videos[]  or  channels[].years[].videos[]
    if "channels" in data:
        videos_dict = {}
        for channel in data["channels"]:
            if "years" in channel:
                for year_data in channel["years"]:
                    for video in year_data.get("videos", []):
                        videos_dict[video["videoId"]] = video
            elif "videos" in channel:
                for video in channel["videos"]:
                    videos_dict[video["videoId"]] = video
        return videos_dict

    # videos[]
    if "videos" in data:
        return {video["videoId"]: video for video in data["videos"]}

    # {videoId: video}
    return data


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(progress: dict) -> None:
    os.makedirs("downloads", exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def download_video(video_id: str, url: str, download_folder: str) -> tuple[bool, str | None]:
    output_path = os.path.join(download_folder, f"{video_id}.mp4")
    if os.path.exists(output_path):
        return True, None

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--extractor-args", "youtube:player_client=android",  # avoids 403
        "-f", "best[height<=720]/best",
        "--postprocessor-args", "ffmpeg:-vf fps=10",          # 10 fps
        "--no-playlist",
        "--retries", "10",
        "--fragment-retries", "10",
        "--socket-timeout", "30",
        "-o", output_path,
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            return True, None
        error_msg = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown error"
        return False, error_msg
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, "yt-dlp is not installed"
    except Exception as e:
        return False, str(e)


def main() -> None:
    videos_file = sys.argv[1] if len(sys.argv) > 1 else "output/shorts_collection.json"
    Path(DOWNLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

    videos = load_videos(videos_file)
    progress = load_progress()
    completed_set = set(progress["completed"])

    to_download = [(vid, data) for vid, data in videos.items() if vid not in completed_set]
    print(f"Total {len(videos)} | done {len(completed_set)} | to download {len(to_download)}")

    for idx, (video_id, video_data) in enumerate(to_download, 1):
        title = (
            video_data.get("title")
            or video_data.get("details", {}).get("snippet", {}).get("title")
            or "(no title)"
        )
        print(f"[{idx}/{len(to_download)}] {video_id} - {title}")

        success, error = download_video(video_id, video_data.get("url"), DOWNLOAD_FOLDER)
        if success:
            progress["completed"].append(video_id)
            if video_id in progress["failed"]:
                progress["failed"].remove(video_id)
        else:
            print(f"  failed: {error}")
            if video_id not in progress["failed"]:
                progress["failed"].append(video_id)

        save_progress(progress)
        if idx < len(to_download):
            time.sleep(5)

    print(f"Done. success {len(progress['completed'])} / failed {len(progress['failed'])}")


if __name__ == "__main__":
    main()
