"""Stage 2 of dataset building: score candidates with a multimodal LLM judge.

Original: "좋아요 점수 - 관련성 점수" section of the
"댓글_시간_노출_편향_보정과 SFT" notebook (Colab, A100).

Input   candidates JSON from dataset_building.create_candidate_dataset()
        [{"video_id", "candidates": [{"text", "final_score", "like_count", ...}]}]
Assets  {asset_root}/{video_id}/frames/*.jpg and audio.wav
        (made by training.qwen_omni_lora.prepare_training_assets)
Output  [{"video_id", "selected_comments": [{"text", "scores": {...}}]}]

The judge sees up to 8 frames and the first 30 s of audio, and scores comments
10 at a time. Final selection sorts by (relevance, final_score) and keeps 10.
The job was split across 3 Colab workers by index range.

Not runnable here: needs a GPU, the private video assets and model weights.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from src.dataset_building.build_sft_dataset import select_final_comments
from src.preprocessing.media_extraction import load_media_for_qwen
from src.relevance_scoring.prompts import (
    RELEVANCE_CHUNK_SIZE,
    build_relevance_judge_prompt,
    parse_relevance_scores,
)

JUDGE_MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


@dataclass(frozen=True)
class JudgeRunConfig:
    input_path: str = "data/dataset_jsons/time_decay+scale_decay_candidates.json"
    output_dir: str = "data/dataset_jsons"
    asset_root_dir: str = "data/train_assets"
    final_selection_size: int = 10
    worker_id: int = 1
    total_workers: int = 3
    max_frames: int = 8
    audio_seconds: float = 30.0


class QwenJudge:
    def __init__(self, model_path: str = JUDGE_MODEL_PATH, asset_root_dir: str = "data/train_assets"):
        self.asset_root_dir = asset_root_dir
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).eval()
        self.model.disable_talker()  # text output only

    def _generate(self, images, audios, content_media, prompt: str) -> str:
        content = content_media + [{"type": "text", "text": prompt}]
        text_prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_prompt],
            images=images if images else None,
            audio=audios if audios else None,
            sampling_rate=16000,
            return_tensors="pt",
            padding=False,
        )
        inputs = {
            k: v.to(self.model.device, dtype=torch.bfloat16) if k in ("pixel_values", "input_features")
            else v.to(self.model.device)
            for k, v in inputs.items() if v is not None
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,   # long enough for 10 "Reason -> Score" lines
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                return_dict_in_generate=False,
            )
        output_ids = outputs[0] if isinstance(outputs, tuple) else outputs
        return self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]

    def evaluate(self, video_id: str, candidates: list[dict], max_frames: int = 8, audio_seconds: float = 30.0) -> list[dict]:
        """Add `relevance_score` (0-100) to every candidate. Unparsed -> 0."""
        asset_dir = os.path.join(self.asset_root_dir, video_id)
        frames_dir = os.path.join(asset_dir, "frames")
        if not os.path.exists(frames_dir):
            return candidates

        images, audios, content_media = load_media_for_qwen(
            frames_dir, os.path.join(asset_dir, "audio.wav"),
            max_frames=max_frames, audio_duration=audio_seconds,
        )
        if not images:
            return candidates

        for i in range(0, len(candidates), RELEVANCE_CHUNK_SIZE):
            chunk = candidates[i:i + RELEVANCE_CHUNK_SIZE]
            prompt = build_relevance_judge_prompt([c["text"] for c in chunk])
            try:
                gen_text = self._generate(images, audios, content_media, prompt)
                for c, score in zip(chunk, parse_relevance_scores(gen_text, len(chunk))):
                    c["relevance_score"] = score
            except Exception as e:
                print(f"Judge error: {e}")
                for c in chunk:
                    c["relevance_score"] = 0.0

        return candidates


def worker_slice(data: list, worker_id: int, total_workers: int) -> list:
    chunk_size = math.ceil(len(data) / total_workers)
    start_idx = (worker_id - 1) * chunk_size
    return data[start_idx:min(start_idx + chunk_size, len(data))]


def run_worker(cfg: JudgeRunConfig = JudgeRunConfig()) -> None:
    with open(cfg.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    my_data = worker_slice(data, cfg.worker_id, cfg.total_workers)
    output_path = os.path.join(cfg.output_dir, f"Score+Relevance_dataset_{cfg.worker_id}.json")

    judge = QwenJudge(asset_root_dir=cfg.asset_root_dir)
    final_dataset = []

    for n, item in enumerate(my_data, 1):
        scored = judge.evaluate(item["video_id"], item["candidates"], cfg.max_frames, cfg.audio_seconds)
        final_dataset.append({
            "video_id": item["video_id"],
            "selected_comments": select_final_comments(scored, cfg.final_selection_size),
        })
        if n % 10 == 0:  # checkpoint every 10 videos
            _write_json(output_path, final_dataset)

    _write_json(output_path, final_dataset)
    print(f"Saved: {output_path}")


def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_worker()
