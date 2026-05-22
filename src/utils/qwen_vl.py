"""Shared lightweight Qwen2.5-VL helpers for prompt/input construction."""

from __future__ import annotations

from typing import Any, Dict

MODEL_NAMES = {
    "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
}


def patch_qwen_visual_sdpa(model) -> None:
    """Patch the Qwen visual tower to use SDPA where available."""
    if not hasattr(model, "visual"):
        return
    if hasattr(model.visual, "config"):
        model.visual.config._attn_implementation = "sdpa"
    for block in getattr(model.visual, "blocks", []):
        if hasattr(block, "attn"):
            block.attn._attn_implementation = "sdpa"


def build_generation_inputs(
    processor,
    sample: Dict[str, Any],
    prompt_text: str,
    device: str,
) -> Dict[str, Any]:
    """Build generation-time multimodal inputs for a user prompt."""
    content = []
    for img in sample["images"]:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[img for img in sample["images"]],
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def build_teacher_forced_inputs(
    processor,
    sample: Dict[str, Any],
    prompt_text: str,
    assistant_text: str,
    device: str,
) -> Dict[str, Any]:
    """Build teacher-forced multimodal inputs with the assistant target appended."""
    content = []
    for img in sample["images"]:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt_text})
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": str(assistant_text)},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = processor(
        text=[text],
        images=[img for img in sample["images"]],
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}
