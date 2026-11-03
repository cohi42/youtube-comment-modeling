"""YouTube Shorts + comment collector.

Original: comment_collect/youtube_shorts_scraper.py (local script).

Flow
  1. Selenium opens each channel's /shorts tab, sorts by "인기순", and scrolls
     until TOP_N_VIDEOS Shorts (<= 60s) are found.
  2. Manually listed video IDs are added and grouped by channel.
  3. YouTube Data API v3 collects every top-level comment (order=time).
  4. API keys rotate automatically when a quota error occurs.
  5. Everything is written to one nested JSON:
     {collectedAt, channels[{channelId, ..., videos[{videoId, ..., comments[]}]}]}

Public-repo changes
  - Real API keys are masked as "[API Key]".
  - The channel handles and video IDs used in the project are not listed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============================================================
# Settings
# ============================================================

# (1) YouTube API keys (several can be registered)
API_KEYS: List[str] = [
    "[API Key]",
    "[API Key]",
    "[API Key]",
    "[API Key]",
]

# (2) ChromeDriver path
CHROMEDRIVER_PATH = os.environ.get("CHROMEDRIVER_PATH", "./chromedriver")

# (3) Channel handles to collect (project list omitted)
TARGET_CHANNELS: List[str] = [
    # "@channel_handle",
]

# (4) Video IDs added manually (project list omitted)
ADDITIONAL_VIDEO_IDS: List[str] = [
    # "VIDEO_ID",
]

# (5) Video IDs to exclude
EXCLUDED_VIDEO_IDS: List[str] = []

# (6) Other settings
TOP_N_VIDEOS = 20      # top N Shorts per channel
HEADLESS_MODE = True
API_DELAY = 0.3        # seconds between API calls
SCROLL_DELAY = 1       # seconds between scrolls

YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_COMMENT_THREADS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


# ============================================================
# API key rotation
# ============================================================

class APIKeyManager:
    """Hold several API keys and switch to the next one on quota errors."""

    def __init__(self, api_keys: List[str]):
        self.api_keys = [k.strip() for k in api_keys if k and k.strip()]
        self.current_index = 0
        self.failed_keys: set[str] = set()

        if not self.api_keys:
            raise ValueError("At least one API key is required.")

    def get_current_key(self) -> str:
        if len(self.failed_keys) >= len(self.api_keys):
            raise Exception("All API keys are exhausted.")
        return self.api_keys[self.current_index]

    def mark_as_failed(self) -> None:
        current_key = self.api_keys[self.current_index]
        self.failed_keys.add(current_key)
        logging.warning(f"API key #{self.current_index + 1} unavailable - switching")

        for _ in range(len(self.api_keys)):
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            if self.api_keys[self.current_index] not in self.failed_keys:
                logging.info(f"Switched to API key #{self.current_index + 1}")
                return

        raise Exception("All API keys are exhausted.")

    @staticmethod
    def is_quota_error(error: Exception) -> bool:
        error_str = str(error).lower()
        quota_keywords = ["quota", "quotaexceeded", "limit exceeded", "rate limit", "too many requests"]
        return any(keyword in error_str for keyword in quota_keywords)


# ============================================================
# Helpers
# ============================================================

def extract_video_id(url: str) -> Optional[str]:
    patterns = [r"/shorts/([A-Za-z0-9_-]+)", r"/watch\?v=([A-Za-z0-9_-]+)", r"v=([A-Za-z0-9_-]+)"]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def parse_iso_duration(duration: str) -> int:
    """ISO 8601 duration (PT1M5S) -> seconds."""
    if not duration:
        return 0
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


# ============================================================
# YouTube Data API
# ============================================================

def get_video_details(video_id: str, api_manager: APIKeyManager) -> Optional[Dict]:
    """videos.list -> snippet, statistics, duration."""
    for _ in range(len(api_manager.api_keys)):
        params = {
            "part": "snippet,statistics,contentDetails",
            "id": video_id,
            "key": api_manager.get_current_key(),
        }
        try:
            response = requests.get(YOUTUBE_VIDEOS_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data.get("items"):
                return None

            item = data["items"][0]
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})
            content_details = item.get("contentDetails", {})

            return {
                "videoId": video_id,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "publishedAt": snippet.get("publishedAt", ""),
                "viewCount": int(statistics.get("viewCount", 0)),
                "likeCount": int(statistics.get("likeCount", 0)),
                "commentCount": int(statistics.get("commentCount", 0)),
                "duration": parse_iso_duration(content_details.get("duration", "PT0S")),
                "channelId": snippet.get("channelId", ""),
                "channelTitle": snippet.get("channelTitle", ""),
            }
        except Exception as e:
            if api_manager.is_quota_error(e):
                logging.warning(f"Quota exceeded: {video_id}")
                api_manager.mark_as_failed()
            else:
                logging.debug(f"API error ({video_id}): {e}")
                return None
    return None


def get_all_comments(video_id: str, api_manager: APIKeyManager) -> List[Dict]:
    """commentThreads.list over every page, oldest first."""
    all_comments: List[Dict] = []
    next_page_token = None

    while True:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "order": "time",
            "key": api_manager.get_current_key(),
            "textFormat": "plainText",
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            response = requests.get(YOUTUBE_COMMENT_THREADS_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                top_comment = item["snippet"]["topLevelComment"]["snippet"]
                all_comments.append({
                    "commentId": item["id"],
                    "author": top_comment.get("authorDisplayName", ""),
                    "text": top_comment.get("textDisplay", ""),
                    "likeCount": top_comment.get("likeCount", 0),
                    "publishedAt": top_comment.get("publishedAt", ""),
                    "updatedAt": top_comment.get("updatedAt", ""),
                })

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break
            time.sleep(API_DELAY)

        except Exception as e:
            if api_manager.is_quota_error(e):
                logging.warning(f"Quota exceeded while collecting comments: {video_id}")
                try:
                    api_manager.mark_as_failed()
                except Exception:
                    logging.error(f"Stopped: {video_id} ({len(all_comments)} comments collected)")
                    break
            else:
                logging.debug(f"Comment error ({video_id}): {e}")
                break

    return all_comments


# ============================================================
# Selenium: popular Shorts per channel
# ============================================================

def collect_shorts_top_by_popularity(
    channel_handle: str,
    chromedriver_path: str,
    api_manager: APIKeyManager,
    top_n: int,
    excluded_ids: set,
    headless: bool,
) -> List[Dict]:
    logging.info(f"Channel: {channel_handle}")

    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    for arg in ("--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                "--window-size=1920,1080", "--lang=ko-KR"):
        chrome_options.add_argument(arg)

    driver = webdriver.Chrome(service=ChromeService(executable_path=chromedriver_path), options=chrome_options)

    videos: List[Dict] = []
    seen_ids: set[str] = set()

    try:
        driver.get(f"https://www.youtube.com/{channel_handle}/shorts")
        time.sleep(3)

        # Click the "인기순" (popular) chip if present
        try:
            popular_xpath = (
                "//div[@id='chip-shape-container']"
                "//button[.//div[contains(normalize-space(.), '인기순')]]"
            )
            popular_btn = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, popular_xpath))
            )
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", popular_btn)
            try:
                popular_btn.click()
            except Exception:
                driver.execute_script("arguments[0].click();", popular_btn)
            time.sleep(2)
        except Exception:
            logging.info("  No popular-sort chip - using default order")

        no_new_count = 0
        max_no_new = 5

        while len(videos) < top_n and no_new_count < max_no_new:
            links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/shorts/']")
            found_new = False

            for link in links:
                if len(videos) >= top_n:
                    break
                try:
                    href = link.get_attribute("href")
                    vid = extract_video_id(href) if href else None
                    if not vid or vid in seen_ids or vid in excluded_ids:
                        continue
                    seen_ids.add(vid)

                    info = get_video_details(vid, api_manager)
                    if not info or info["duration"] > 60:  # Shorts only
                        continue

                    videos.append(info)
                    found_new = True
                    time.sleep(API_DELAY)
                except Exception:
                    continue

            no_new_count = 0 if found_new else no_new_count + 1
            if len(videos) < top_n and no_new_count >= max_no_new:
                logging.info(f"  {channel_handle}: only {len(videos)} found")
                break

            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(SCROLL_DELAY)
    finally:
        driver.quit()

    logging.info(f"  {channel_handle}: {len(videos)} collected")
    return videos


def process_additional_videos(
    video_ids: List[str],
    api_manager: APIKeyManager,
    excluded_ids: set,
) -> Dict[str, List[Dict]]:
    """Fetch manually listed videos and group them by channel."""
    channel_videos: Dict[str, List[Dict]] = defaultdict(list)
    for vid in video_ids:
        if vid in excluded_ids:
            continue
        info = get_video_details(vid, api_manager)
        if not info:
            logging.warning(f"  No info: {vid}")
            continue
        channel_videos[info["channelId"]].append(info)
        time.sleep(API_DELAY)
    return dict(channel_videos)


def merge_videos_by_channel(
    channel_videos: Dict[str, List[Dict]],
    additional_videos: Dict[str, List[Dict]],
) -> Dict[str, List[Dict]]:
    all_videos: Dict[str, List[Dict]] = defaultdict(list)
    for source in (channel_videos, additional_videos):
        for channel_id, videos in source.items():
            all_videos[channel_id].extend(videos)
    for channel_id in all_videos:
        all_videos[channel_id].sort(key=lambda x: x["viewCount"], reverse=True)
    return dict(all_videos)


# ============================================================
# Main
# ============================================================

def main(output_dir: str = "output") -> None:
    if not os.path.exists(CHROMEDRIVER_PATH):
        logging.error(f"ChromeDriver not found: {CHROMEDRIVER_PATH}")
        return
    if not TARGET_CHANNELS and not ADDITIONAL_VIDEO_IDS:
        logging.error("Add at least one channel or video ID.")
        return

    api_manager = APIKeyManager(API_KEYS)
    excluded_ids = set(EXCLUDED_VIDEO_IDS)

    # Step 1: popular Shorts per channel
    channel_videos: Dict[str, List[Dict]] = {}
    for channel in TARGET_CHANNELS:
        videos = collect_shorts_top_by_popularity(
            channel, CHROMEDRIVER_PATH, api_manager, TOP_N_VIDEOS, excluded_ids, HEADLESS_MODE
        )
        if videos:
            channel_videos[videos[0]["channelId"]] = videos

    # Step 2: manually added videos
    additional_videos = (
        process_additional_videos(ADDITIONAL_VIDEO_IDS, api_manager, excluded_ids)
        if ADDITIONAL_VIDEO_IDS else {}
    )

    # Step 3: merge and sort by views
    all_videos = merge_videos_by_channel(channel_videos, additional_videos)

    # Step 4: collect comments
    channels_data = []
    for channel_id, videos in all_videos.items():
        if not videos:
            continue

        channel_handle = ""
        for handle in TARGET_CHANNELS:
            if any(v["channelId"] == channel_id for v in channel_videos.get(channel_id, [])):
                channel_handle = handle
                break

        videos_data = []
        for rank, video in enumerate(videos, 1):
            vid = video["videoId"]
            comments = get_all_comments(vid, api_manager)
            logging.info(f"  #{rank} {video['title'][:50]} (views {video['viewCount']:,}, comments {len(comments)})")

            videos_data.append({
                "rank": rank,
                "videoId": vid,
                "title": video["title"],
                "url": f"https://www.youtube.com/shorts/{vid}",
                "publishedAt": video["publishedAt"],
                "viewCount": video["viewCount"],
                "likeCount": video["likeCount"],
                "duration": video["duration"],
                "originalCommentCount": video["commentCount"],
                "collectedCommentCount": len(comments),
                "comments": comments,
            })

        channels_data.append({
            "channelId": channel_id,
            "channelTitle": videos[0]["channelTitle"],
            "channelHandle": channel_handle,
            "totalVideos": len(videos),
            "videos": videos_data,
        })

    # Step 5: save one JSON
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"shorts_collection_{timestamp}.json")

    final_data = {
        "collectedAt": datetime.now().isoformat(),
        "topN": TOP_N_VIDEOS,
        "totalChannels": len(channels_data),
        "channels": channels_data,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
