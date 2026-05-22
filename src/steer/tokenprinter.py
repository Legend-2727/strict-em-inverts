"""
Token Printer
=============

Utility for inspecting the token sequence fed into a Qwen2.5-VL model.
For each position the function prints:

  - Text / special tokens  →  position, token-id, decoded string
  - Image patch tokens     →  position, token-id, (row, col) in the patch grid

The Qwen2.5-VL processor packs image patches as ``<|image_pad|>`` tokens
(id 151655) between ``<|vision_start|>`` (id 151652) and
``<|vision_end|>`` (id 151653).  The spatial layout is taken from the
``image_grid_thw`` tensor that the processor returns alongside
``input_ids``.
"""

from __future__ import annotations

from typing import Optional
import torch


# Default special-token IDs for Qwen2.5-VL
_VISION_START_ID: int = 151652
_VISION_END_ID: int = 151653
_IMAGE_PAD_ID: int = 151655


def _get_grid_thw(inputs) -> Optional[torch.Tensor]:
    """Return the ``image_grid_thw`` tensor from the processor output batch."""
    for attr in ("image_grid_thw",):
        val = getattr(inputs, attr, None)
        if val is None:
            val = inputs.get(attr) if hasattr(inputs, "get") else None
        if val is not None:
            return val
    return None


def print_tokens(
    inputs,
    processor,
    image_token_id: int = _IMAGE_PAD_ID,
    vision_start_id: int = _VISION_START_ID,
    vision_end_id: int = _VISION_END_ID,
    batch_idx: int = 0,
) -> None:
    """
    Print every token in the input sequence with human-readable annotations.

    For image patch tokens the grid coordinate ``(row, col)`` within the
    spatial patch grid is shown.  The patch grid dimensions are derived from
    ``image_grid_thw``:  effective height = ``grid_h // 2``, effective
    width = ``grid_w // 2`` (Qwen2.5-VL merges 2×2 patches).

    Args:
        inputs:          Output of ``processor(...)`` - a ``BatchFeature``
                         (or any mapping / object) that contains ``input_ids``
                         and, for visual inputs, ``image_grid_thw``.
        processor:       The ``AutoProcessor`` used to prepare the inputs.
                         Its ``tokenizer`` attribute is used for decoding.
        image_token_id:  Token id that marks image patch positions
                         (default: 151655 for Qwen2.5-VL).
        vision_start_id: Token id for ``<|vision_start|>`` (default: 151652).
        vision_end_id:   Token id for ``<|vision_end|>``   (default: 151653).
        batch_idx:       Which example in the batch to inspect (default: 0).
    """
    # ── Retrieve input_ids ───────────────────────────────────────────────────
    input_ids: torch.Tensor = inputs["input_ids"][batch_idx]   # [seq_len]

    # ── Retrieve spatial grid info ───────────────────────────────────────────
    image_grid_thw = _get_grid_thw(inputs)  # [num_images, 3] or None

    # Build a list of (h_tokens, w_tokens) per image
    grid_dims: list[tuple[int, int]] = []
    if image_grid_thw is not None:
        for thw in image_grid_thw:
            t, h, w = thw.tolist()
            grid_dims.append((int(h) // 2, int(w) // 2))

    # ── Tokenizer for decoding ───────────────────────────────────────────────
    tokenizer = getattr(processor, "tokenizer", processor)

    # ── Print header ─────────────────────────────────────────────────────────
    total = input_ids.shape[0]
    n_image = int((input_ids == image_token_id).sum().item())
    n_text = total - n_image
    print(f"\n{'─'*60}")
    print(f"  Token sequence:  {total} tokens total  "
          f"({n_text} text/special,  {n_image} image-patch)")
    if grid_dims:
        desc = ", ".join(f"{h}×{w}" for h, w in grid_dims)
        print(f"  Image patch grids: {desc}")
    print(f"{'─'*60}")
    print(f"  {'pos':>5}  {'id':>7}  {'type':<12}  detail")
    print(f"{'─'*60}")

    # ── Walk through the sequence ────────────────────────────────────────────
    image_img_idx = 0   # which image we are currently in
    image_patch_ctr = 0  # patch counter within the current image
    inside_image = False

    for pos, tok_id_tensor in enumerate(input_ids):
        tok_id = int(tok_id_tensor.item())

        # --- vision boundary tokens ---
        if tok_id == vision_start_id:
            inside_image = True
            image_patch_ctr = 0
            decoded = tokenizer.decode([tok_id])
            print(f"  {pos:>5}  {tok_id:>7}  {'vision-start':<12}  {decoded!r}")
            continue

        if tok_id == vision_end_id:
            inside_image = False
            decoded = tokenizer.decode([tok_id])
            print(f"  {pos:>5}  {tok_id:>7}  {'vision-end':<12}  {decoded!r}")
            image_img_idx += 1
            continue

        # --- image patch token ---
        if tok_id == image_token_id and inside_image:
            if image_img_idx < len(grid_dims):
                h_tok, w_tok = grid_dims[image_img_idx]
                row = image_patch_ctr // w_tok
                col = image_patch_ctr % w_tok
                grid_str = f"grid[{row},{col}] (image {image_img_idx}, patch {image_patch_ctr})"
            else:
                grid_str = f"patch {image_patch_ctr}"
            print(f"  {pos:>5}  {tok_id:>7}  {'image-patch':<12}  {grid_str}")
            image_patch_ctr += 1
            continue

        # --- ordinary text / special token ---
        decoded = tokenizer.decode([tok_id])
        print(f"  {pos:>5}  {tok_id:>7}  {'text':<12}  {decoded!r}")

    print(f"{'─'*60}\n")
