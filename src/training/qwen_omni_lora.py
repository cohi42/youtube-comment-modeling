"""Qwen3-Omni LoRA/SFT on video frames + audio -> comment.

Original: "모델 학습" section of the
"Qwen_Qwen3_Omni_30B_A3B_Instruct_+_Quantization" notebook (Colab, A100).

Steps
  0. prepare_training_assets(): extract frames/audio for every video once,
     zip them, and unzip on the training runtime
  1. Load Qwen3-Omni-30B-A3B in BF16 (no quantization), disable the talker
  2. Patch get_input_embeddings (the MoE wrapper does not expose it to PEFT)
  3. Gradient checkpointing + input grads
  4. LoRA r=64, alpha=16 on q/k/v/o projections
  5. Route forward() to the Thinker (text-generating sub-model)
  6. Multimodal dataset + collator, HF Trainer, 1 epoch

Not runnable here: needs an 80 GB GPU, the private video assets and model
weights. Paths below are placeholders for the original Google Drive layout.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import random
import shutil
import types
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from src.preprocessing.media_extraction import TRAINING_ASSETS, extract_media, load_media_for_qwen

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Prompts were sampled at random per example
PROMPT_LIST = [
    "이 영상에 달릴법한 재치있는 댓글을 작성해줘.",
    "재미있는 댓글 써줘.",
]


@dataclass(frozen=True)
class TrainConfig:
    json_file_path: str = "data/dataset_jsons/final_grouped_sft_data.json"
    asset_root_dir: str = "data/train_assets"
    output_dir: str = "outputs/qwen_lora_result"
    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # Trainer
    batch_size: int = 4
    grad_accum: int = 1
    epochs: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    save_steps: int = 100
    max_frames: int = 8


# ============================================================
# 0. Asset preparation (video -> frames + audio -> zip)
# ============================================================

def prepare_training_assets(
    source_video_dir: str = "data/videos",
    local_work_dir: str = "data/processed_assets",
    zip_dest_dir: str = "data",
    zip_name: str = "processed_dataset",
) -> str:
    """Extract {video_id}/frames/*.jpg + audio.wav for every mp4 and zip the result.

    Extraction ran once on local Colab disk; the zip was moved to Drive and
    unzipped into the training runtime, which was much faster than reading
    thousands of small files from Drive.
    """
    if os.path.exists(local_work_dir):
        shutil.rmtree(local_work_dir)
    os.makedirs(local_work_dir)

    video_files = glob.glob(os.path.join(source_video_dir, "*.mp4"))
    for video_path in tqdm(video_files):
        file_id = os.path.splitext(os.path.basename(video_path))[0]
        extract_media(video_path, os.path.join(local_work_dir, file_id), TRAINING_ASSETS)

    zip_path = shutil.make_archive(zip_name, "zip", local_work_dir)
    final_dest = os.path.join(zip_dest_dir, f"{zip_name}.zip")
    shutil.move(zip_path, final_dest)
    return final_dest


def unpack_training_assets(zip_path: str, asset_root_dir: str) -> None:
    if not os.path.exists(asset_root_dir):
        shutil.unpack_archive(zip_path, asset_root_dir)


# ============================================================
# 1-5. Model
# ============================================================

def load_base_model(model_path: str = MODEL_PATH):
    """BF16 full load on one GPU. SDPA attention instead of Flash Attention."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduce fragmentation

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.disable_talker()          # no speech output needed
    model.config.use_cache = False
    return model


def patch_model_embeddings(model) -> None:
    """Expose the Thinker's token embedding through get_input_embeddings().

    PEFT and gradient checkpointing call model.get_input_embeddings(), which the
    Omni MoE wrapper does not implement. Find `embed_tokens` (or the largest
    nn.Embedding) inside the Thinker and bind it.
    """
    target_module = model.thinker if hasattr(model, "thinker") else model
    found_embedding = None

    for name, module in target_module.named_modules():
        if isinstance(module, nn.Embedding) and "embed_tokens" in name:
            found_embedding = module
            break

    if found_embedding is None:
        max_size = 0
        for _, module in target_module.named_modules():
            if isinstance(module, nn.Embedding) and module.num_embeddings > max_size:
                max_size = module.num_embeddings
                found_embedding = module

    if found_embedding is None:
        raise ValueError("Embedding layer not found")

    model.get_input_embeddings = types.MethodType(lambda self: found_embedding, model)
    model.set_input_embeddings = types.MethodType(lambda self, value: None, model)


def enable_memory_saving(model) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()


def attach_lora(model, cfg: TrainConfig):
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def get_actual_model(peft_model):
    """Unwrap PeftModel -> LoraModel -> Qwen3OmniMoeForConditionalGeneration."""
    current, visited = peft_model, set()
    for _ in range(10):
        if id(current) in visited:
            break
        visited.add(id(current))
        if hasattr(current, "base_model") and current.base_model is not current:
            current = current.base_model
        elif hasattr(current, "model") and current.model is not current:
            current = current.model
        else:
            break
    return current


def route_forward_to_thinker(peft_model) -> None:
    """Trainer calls model(**batch); send it to the Thinker with audio renamed.

    The collator emits `input_audio_features`; the Thinker expects `input_features`.
    """
    actual = get_actual_model(peft_model)
    if not hasattr(actual, "thinker"):
        raise ValueError("Thinker module not found")
    thinker_model = actual.thinker

    def custom_forward(input_ids=None, attention_mask=None, pixel_values=None,
                       input_audio_features=None, labels=None, **kwargs):
        return thinker_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            input_features=input_audio_features,
            labels=labels,
            **kwargs,
        )

    peft_model.forward = custom_forward


# ============================================================
# 6. Data
# ============================================================

class QwenMultimodalDataset(Dataset):
    """One example per (video, selected comment): frames + audio + prompt -> comment."""

    def __init__(self, json_path: str, asset_dir: str, processor, max_frames: int = 8):
        self.processor = processor
        self.asset_dir = asset_dir
        self.max_frames = max_frames

        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.flattened_data = []
        for item in raw_data:
            vid = item.get("video_id", item.get("input"))
            if not vid:
                continue
            if "selected_comments" in item:
                for c in item["selected_comments"]:
                    self.flattened_data.append({"video_id": vid, "target_text": c["text"]})
            elif "output" in item:
                self.flattened_data.append({"video_id": vid, "target_text": item["output"]})

    def __len__(self) -> int:
        return len(self.flattened_data)

    def __getitem__(self, idx: int) -> dict:
        data = self.flattened_data[idx]
        asset = os.path.join(self.asset_dir, data["video_id"])

        images, audios, user_content = load_media_for_qwen(
            os.path.join(asset, "frames"), os.path.join(asset, "audio.wav"), max_frames=self.max_frames
        )
        user_content.append({"type": "text", "text": random.choice(PROMPT_LIST)})

        text_prompt = self.processor.apply_chat_template(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": data["target_text"]}]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=[text_prompt],
            images=images if images else None,
            audio=audios if audios else None,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
            if "image_grid_thw" in inputs:
                result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
            elif images:
                result["image_grid_thw"] = torch.tensor([[1, 28, 28]] * len(images), dtype=torch.long)
        if "input_audio_features" in inputs:
            result["input_audio_features"] = inputs["input_audio_features"]
        return result


def make_data_collator(processor):
    """Right-pad to the longest sequence. Labels = input_ids, padding = -100."""
    pad_id = processor.tokenizer.pad_token_id

    def data_collator(features: list[dict]) -> dict:
        max_len = max(f["input_ids"].shape[0] for f in features)
        input_ids, masks, labels = [], [], []
        for f in features:
            pad_len = max_len - f["input_ids"].shape[0]
            input_ids.append(torch.cat([f["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
            masks.append(torch.cat([f["attention_mask"], torch.full((pad_len,), 0, dtype=torch.long)]))
            labels.append(torch.cat([f["input_ids"], torch.full((pad_len,), -100, dtype=torch.long)]))

        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels),
        }

        pixel_values = [f["pixel_values"] for f in features if f.get("pixel_values") is not None]
        if pixel_values:
            batch["pixel_values"] = torch.cat([pv.unsqueeze(0) for pv in pixel_values])
            grids = [f["image_grid_thw"] for f in features if f.get("image_grid_thw") is not None]
            if grids:
                batch["image_grid_thw"] = torch.cat([g.unsqueeze(0) if g.dim() == 1 else g for g in grids])

        audio_feats = [f["input_audio_features"] for f in features if f.get("input_audio_features") is not None]
        if audio_feats:
            batch["input_audio_features"] = torch.cat(audio_feats)
        return batch

    return data_collator


class MemoryCleanupCallback(TrainerCallback):
    """gc + empty_cache every 20 steps (Colab OOM guard)."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        return control


# ============================================================
# Entry point
# ============================================================

def train(cfg: TrainConfig = TrainConfig()) -> None:
    model = load_base_model()
    patch_model_embeddings(model)
    enable_memory_saving(model)
    model = attach_lora(model, cfg)
    route_forward_to_thinker(model)

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    train_dataset = QwenMultimodalDataset(cfg.json_file_path, cfg.asset_root_dir, processor, cfg.max_frames)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,   # keep pixel_values / audio features
        gradient_checkpointing=False,  # already enabled on the model
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        warmup_steps=cfg.warmup_steps,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_data_collator(processor),
    )
    trainer.add_callback(MemoryCleanupCallback())
    trainer.train()

    model.save_pretrained(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    train()
