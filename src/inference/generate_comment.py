"""Generate a comment for a video with the LoRA-tuned Qwen3-Omni.

Original: LoRA inference cells at the end of the
"Qwen_Qwen3_Omni_30B_A3B_Instruct_+_Quantization" notebook (Colab, A100).

  1. Load the base model in BF16 (same dtype as training, no BitsAndBytes)
  2. Apply the same embedding patch as training
  3. Attach the LoRA adapter (checkpoint-1388)
  4. Extract frames (224 px) + audio on the fly, generate with sampling

Example output from the notebook:
  video  "그렇게 맛있다고 극찬들로만 가득한 버거 드디어 먹어봤습니다"
  output "이거 먹고 싶어도 맛집이라서 못먹겠네..."

Not runnable here: needs a GPU, model weights and the trained adapter.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from peft import PeftModel
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from src.preprocessing.media_extraction import LORA_INFERENCE, extract_media, load_media_for_qwen, reset_dir
from src.training.qwen_omni_lora import MODEL_PATH, patch_model_embeddings

DEFAULT_COMMENT_PROMPT = "이 영상에 달릴법한 재치있는 댓글을 작성해줘."


@dataclass(frozen=True)
class GenerationConfig:
    adapter_path: str = "outputs/qwen_lora_result/checkpoint-1388"
    base_model_path: str = MODEL_PATH
    temp_work_dir: str = "outputs/inference_temp"
    max_new_tokens: int = 128
    min_new_tokens: int = 5
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 5


def set_seed(seed_val: int = 4242) -> None:
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def load_lora_model(cfg: GenerationConfig = GenerationConfig()):
    base_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        cfg.base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    base_model.disable_talker()
    patch_model_embeddings(base_model)  # required before loading the adapter

    model = PeftModel.from_pretrained(base_model, cfg.adapter_path)
    model.eval()

    processor = Qwen3OmniMoeProcessor.from_pretrained(cfg.base_model_path)
    # Decoder-only generation continues on the right, so pad on the left
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor


INT_KEYS = ("input_ids", "attention_mask", "image_grid_thw", "video_grid_thw", "feature_attention_mask")
FLOAT_KEYS = ("pixel_values", "input_features", "input_audio_features")


def _to_device(raw_inputs, device) -> dict:
    """Integer tensors -> long, feature tensors -> bf16."""
    inputs = {}
    for key, value in raw_inputs.items():
        if value is None:
            continue
        if key in INT_KEYS:
            inputs[key] = value.to(device=device, dtype=torch.long)
        elif key in FLOAT_KEYS:
            inputs[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            inputs[key] = value.to(device=device)
    return inputs


def generate_comment(
    model,
    processor,
    video_path: str,
    prompt: str = DEFAULT_COMMENT_PROMPT,
    seed: int = 42,
    cfg: GenerationConfig = GenerationConfig(),
) -> str:
    set_seed(seed)

    reset_dir(cfg.temp_work_dir)
    frames_dir, audio_path = extract_media(video_path, cfg.temp_work_dir, LORA_INFERENCE)
    images, audios, content = load_media_for_qwen(frames_dir, audio_path)
    content.append({"type": "text", "text": prompt})

    text_prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    raw_inputs = processor(
        text=[text_prompt],
        images=images if images else None,
        audio=audios if audios else None,
        sampling_rate=16000,
        return_tensors="pt",
        padding=False,  # batch size 1
    )
    inputs = _to_device(raw_inputs, model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=cfg.min_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            pad_token_id=processor.tokenizer.eos_token_id,
            return_dict_in_generate=False,
        )

    output_ids = outputs[0] if isinstance(outputs, tuple) else outputs
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:].cpu().tolist()
    if not generated_ids:
        return "[error] empty generation"
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


if __name__ == "__main__":
    import sys

    video = sys.argv[1] if len(sys.argv) > 1 else "data/videos/sample.mp4"
    if not os.path.exists(video):
        raise SystemExit(f"Video not found: {video}")
    model, processor = load_lora_model()
    print(generate_comment(model, processor, video))
