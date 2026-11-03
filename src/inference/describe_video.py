"""Describe each video in text with 4-bit Qwen3-Omni (the "Text description" input).

Original: "비디오 추론" section of the
"Qwen_Qwen3_Omni_30B_A3B_Instruct_+_Quantization" notebook (Colab).

The 30B model is loaded in 4-bit NF4 (BitsAndBytes) so it fits next to the
video processing. For every video, frames (448 px, 4-8) and audio are extracted
on the fly, a description is generated, and saved as {video_id}_result.txt.
Finished videos are skipped, so a disconnected Colab session can resume.

These descriptions were the input for the "Text description" rows of the
experiment table (docs/experiments.md).

Credentials: the Hugging Face token used in the notebook is masked as "[API Key]".
"""

from __future__ import annotations

import glob
import os
import traceback
from dataclasses import dataclass

import torch
from transformers import BitsAndBytesConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from src.preprocessing.media_extraction import DESCRIPTION_INFERENCE, extract_media, load_media_for_qwen, reset_dir

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

SYSTEM_PROMPT = "You are a helpful AI assistant capable of analyzing video content."
USER_PROMPT = (
    "Describe what is happening in this video based on the images and audio provided. "
    "Focus on the main actions and events. "
    "Ignore any overlay text, UI elements, or subtitles at the top/bottom of the screen."
)


@dataclass(frozen=True)
class DescribeConfig:
    video_root_dir: str = "data/videos"
    result_dir: str = "outputs/video_descriptions"
    temp_work_dir: str = "outputs/temp_workspace"


def hf_login() -> None:
    from huggingface_hub import login

    login("[API Key]")


def load_quantized_model(model_path: str = MODEL_PATH):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,  # allow CPU offload of fp32 modules
    )
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        attn_implementation="sdpa",
        quantization_config=quantization_config,
    )
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    return model, processor


def describe(model, processor, images, audios, content) -> str:
    content = content + [{"type": "text", "text": USER_PROMPT}]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt],
        images=images if images else None,
        audio=audios if audios else None,
        sampling_rate=16000,
        return_tensors="pt",
    ).to(device="cuda", dtype=torch.float16)

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=pad_id,
            return_dict_in_generate=False,
        )
    if isinstance(output, tuple):
        output = output[0]

    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output)]
    return processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def process_all_videos(model, processor, cfg: DescribeConfig = DescribeConfig()) -> None:
    os.makedirs(cfg.result_dir, exist_ok=True)

    video_files = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
        video_files.extend(glob.glob(os.path.join(cfg.video_root_dir, ext)))

    for idx, video_path in enumerate(video_files, 1):
        file_id = os.path.splitext(os.path.basename(video_path))[0]
        result_txt_path = os.path.join(cfg.result_dir, f"{file_id}_result.txt")
        if os.path.exists(result_txt_path):
            continue  # resume support

        print(f"[{idx}/{len(video_files)}] {file_id}")
        reset_dir(cfg.temp_work_dir)
        frames_dir, audio_path = extract_media(video_path, cfg.temp_work_dir, DESCRIPTION_INFERENCE)
        images, audios, content = load_media_for_qwen(frames_dir, audio_path)

        if not images and not audios:
            print("  no content extracted - skipped")
            continue

        try:
            response_text = describe(model, processor, images, audios, content)
            with open(result_txt_path, "w", encoding="utf-8") as f:
                f.write(response_text)
        except Exception as e:
            print(f"  inference error: {e}")
            traceback.print_exc()

        reset_dir(cfg.temp_work_dir)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    hf_login()
    model, processor = load_quantized_model()
    process_all_videos(model, processor)
