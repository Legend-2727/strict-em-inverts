import torch
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image
from .steermanager import SteerManager


class AttentionManager(SteerManager):
    def _extract_attn_hook(self, layer_name: str, module, input, output):
        """Extracts attention weights instead of hidden states.

        Returns modified output with attention weights replaced by ``None``
        so the GPU tensor (~2.8 GiB per layer) is freed immediately.
        """
        # Qwen's self_attn output tuple: (attn_output, attn_weights, past_key_values)
        if isinstance(output, tuple) and len(output) > 1:
            if output[1] is None:
                raise ValueError(
                    "Attention weights are None. Ensure `output_attentions=True` is set "
                    "in the model config and the attention implementation supports "
                    "returning weights (flash_attention_2 does not).")
            # attn_weights shape: [batch, heads, seq_len, seq_len]
            attn_weights = output[1].detach().cpu().to(torch.float32)
            self.storage[layer_name].append(attn_weights)
            # Free GPU memory by stripping weights from output
            return (output[0], None) + output[2:] if len(output) > 2 else (output[0], None)
        else:
            raise ValueError(
                "Attention weights missing. Pass `output_attentions=True` to the model.")

    def attach_attention_probe(self, layer_name: str):
        """Attaches the modified attention hook.

        Also sets ``model.config.output_attentions = True`` so that
        attention layers actually return their weight matrices instead
        of ``None``.
        """
        self.model.config.output_attentions = True
        layer = self._get_layer(layer_name)
        hook = layer.register_forward_hook(
            partial(self._extract_attn_hook, layer_name)
        )
        self.hooks.append(hook)
        return self


def visualize_qwen_dynamic_attention(
    attn_matrix: torch.Tensor,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    original_image: Image.Image,  # Added parameter
    image_token_id: int = 151655,
    alpha: float = 0.5  # Added opacity control
):
    """
    attn_matrix: Shape [Batch, Heads, Seq_Len, Seq_Len]
    input_ids: Shape [Batch, Seq_Len]
    image_grid_thw: Shape [Batch, 3] containing (Time, Height, Width)
    original_image: PIL Image of the input
    """
    avg_attn = attn_matrix[0].mean(dim=0)
    query_idx = -1
    token_attn = avg_attn[query_idx]

    vision_token_mask = (input_ids[0] == image_token_id)
    vision_attn_1d = token_attn[vision_token_mask]

    grid_t, grid_h, grid_w = image_grid_thw[0].tolist()
    h_tokens = grid_h // 2
    w_tokens = grid_w // 2

    if len(vision_attn_1d) != (h_tokens * w_tokens):
        raise ValueError(
            f"Token count mismatch. Found {len(vision_attn_1d)}, expected {h_tokens * w_tokens}")

    heatmap_2d = vision_attn_1d.view(h_tokens, w_tokens).numpy()

    # Normalize heatmap for better visualization contrast
    heatmap_2d = (heatmap_2d - heatmap_2d.min()) / \
        (heatmap_2d.max() - heatmap_2d.min() + 1e-8)

    # Plotting
    plt.figure(figsize=(8, 8))
    # 1. Show original image
    plt.imshow(original_image)
    # 2. Overlay heatmap with alpha and extent
    plt.imshow(
        heatmap_2d,
        cmap='jet',
        # interpolation='bicubic',
        alpha=alpha,
        # Matches heatmap coordinates to image size
        extent=[0, original_image.width, original_image.height, 0]
    )
    plt.axis('off')
    plt.title(f"Attention Grid: {w_tokens}x{h_tokens}")
    plt.show()


def save_qwen_dynamic_attention(
    attn_matrix: torch.Tensor,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    save_path: str,
    layer_name: str,
    original_image: Image.Image,
    image_token_id: int = 151655,
    alpha: float = 0.5
):
    avg_attn = attn_matrix[0].mean(dim=0)
    query_idx = -1
    token_attn = avg_attn[query_idx]

    vision_token_mask = (input_ids[0] == image_token_id)
    vision_attn_1d = token_attn[vision_token_mask]

    grid_t, grid_h, grid_w = image_grid_thw[0].tolist()
    h_tokens = grid_h // 2
    w_tokens = grid_w // 2

    if len(vision_attn_1d) != (h_tokens * w_tokens):
        raise ValueError(
            f"Token count mismatch. Found {len(vision_attn_1d)}, expected {h_tokens * w_tokens}")

    heatmap_2d = vision_attn_1d.view(h_tokens, w_tokens).cpu().numpy()

    # Normalize heatmap
    heatmap_2d = (heatmap_2d - heatmap_2d.min()) / \
        (heatmap_2d.max() - heatmap_2d.min() + 1e-8)

    # Plot and Save
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.imshow(
        heatmap_2d,
        cmap='jet',
        # interpolation='bicubic',
        alpha=alpha,
        extent=[0, original_image.width, original_image.height, 0]
    )
    plt.axis('off')
    plt.title(f"Attention Map: {layer_name}\nGrid: {w_tokens}x{h_tokens}")

    # Added dpi for higher quality output
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


# =============================================================================
# CLAIM: Cross-Lingual Attention Intervention for Visual Text Steering
# =============================================================================


class CLAIMManager(SteerManager):
    """
    Manager for extracting per-head attention outputs used in the CLAIM pipeline.

    Hooks into three points of each attention layer:
      - ``self_attn`` → captures post-softmax attention weights
      - ``v_proj``    → captures value projection output (for masked output computation)
      - ``o_proj``    → captures pre-projection concatenated per-head attention output

    These enable computing both **standard** and **masked** (vision-only)
    per-head attention outputs required by CLAIM Phases 1 and 2.

    Args:
        model: The VLM model (e.g. ``Qwen2VLForConditionalGeneration``).
        num_heads: Number of attention heads (query heads).
        num_kv_heads: Number of key-value heads (for GQA).
        head_dim: Dimension per attention head.
        layer_name_template: Format string for self_attn layer paths.
    """

    def __init__(
        self,
        model,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_name_template: str = "model.language_model.layers.{}.self_attn",
        token_pos: int = -1,
    ):
        super().__init__(model)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.layer_name_template = layer_name_template
        self.token_pos = token_pos
        # Separate storage per data type
        self.attn_weights_storage: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.value_storage: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.oproj_input_storage: Dict[int, List[torch.Tensor]] = defaultdict(list)

    # ----- hooks -----

    def _attn_weight_hook(self, layer_idx: int, module, input, output):
        """Capture attention weights at ``self.token_pos`` from self_attn.

        Stores only the row at the configured token position — shape
        ``[heads, seq]`` (~0.6 MB) instead of the full ``[batch, heads,
        seq, seq]`` matrix (~2.8 GiB).  The GPU tensor is freed by
        replacing the weights in the output tuple with ``None``.
        """
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            # Slice to [heads, seq] at token_pos — 4,600x smaller than full matrix
            self.attn_weights_storage[layer_idx].append(
                output[1][0, :, self.token_pos, :].detach().cpu().to(torch.float32)
            )
            # Free GPU memory by stripping weights from output
            return (output[0], None) + output[2:] if len(output) > 2 else (output[0], None)

    def _value_hook(self, layer_idx: int, module, input, output):
        """Capture value projection output from v_proj (batch index 0)."""
        self.value_storage[layer_idx].append(
            output[0].detach().cpu().to(torch.float32)
        )

    def _oproj_input_hook(self, layer_idx: int, module, input, output):
        """Capture o_proj input at ``self.token_pos`` only.

        Stores a ``[hidden]`` vector (~14 KB) instead of the full
        ``[batch, seq, hidden]`` tensor (~74 MB).
        """
        self.oproj_input_storage[layer_idx].append(
            input[0][0, self.token_pos, :].detach().cpu().to(torch.float32)
        )

    # ----- probe attachment -----

    def attach_claim_probes(
        self,
        layer_indices: List[int],
        capture_attn_weights: bool = True,
        capture_values: bool = True,
    ):
        """
        Attach extraction hooks for CLAIM at the specified layers.

        Args:
            layer_indices: Decoder layer indices to probe.
            capture_attn_weights: Whether to capture attention weight matrices
                (needed for Phase 1 masked output computation).
            capture_values: Whether to capture value states
                (needed for Phase 1 masked output computation).
        """
        self.model.config.output_attentions = capture_attn_weights
        for idx in layer_indices:
            base = self.layer_name_template.format(idx)

            if capture_attn_weights:
                layer = self._get_layer(base)
                hook = layer.register_forward_hook(
                    partial(self._attn_weight_hook, idx)
                )
                self.hooks.append(hook)

            if capture_values:
                v_layer = self._get_layer(f"{base}.v_proj")
                hook = v_layer.register_forward_hook(
                    partial(self._value_hook, idx)
                )
                self.hooks.append(hook)

            # Always capture o_proj input (needed for both Phase 1 and 2)
            o_layer = self._get_layer(f"{base}.o_proj")
            hook = o_layer.register_forward_hook(
                partial(self._oproj_input_hook, idx)
            )
            self.hooks.append(hook)
        return self

    # ----- output extraction -----

    def get_perhead_output(
        self, layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Get the standard per-head attention output at the configured
        ``token_pos``.

        Returns:
            Tensor of shape ``[num_heads, head_dim]`` or ``None``.
        """
        data = self.oproj_input_storage.get(layer_idx, [])
        if not data:
            return None
        # Already sliced to [hidden] in hook
        return data[0].view(self.num_heads, self.head_dim)

    def get_masked_output(
        self,
        layer_idx: int,
        vision_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute masked attention output using only vision tokens.

        Zeroes out attention to non-vision (text) tokens and renormalises,
        replicating the causal-mask procedure from CLAIM Phase 1.

        Args:
            layer_idx: Decoder layer index.
            vision_mask: Boolean tensor ``[seq_len]`` where ``True`` marks
                vision tokens.

        Returns:
            Tensor of shape ``[num_heads, head_dim]`` or ``None``.
        """
        attn_data = self.attn_weights_storage.get(layer_idx, [])
        val_data = self.value_storage.get(layer_idx, [])
        if not attn_data or not val_data:
            return None

        attn_at_pos = attn_data[0]   # [heads, seq] — already sliced in hook
        values = val_data[0]         # [seq, kv_dim] — batch index 0

        seq_len, _ = values.shape
        # Reshape and expand for GQA: [kv_heads, seq, hd] → [heads, seq, hd]
        values = values.view(seq_len, self.num_kv_heads, self.head_dim)
        values = values.permute(1, 0, 2)  # [kv_heads, seq, hd]
        values = values.repeat_interleave(self.num_kv_groups, dim=0)

        # Apply vision mask and renormalise
        mask = vision_mask.float()
        masked_attn = attn_at_pos * mask.unsqueeze(0)      # [heads, seq]
        attn_sum = masked_attn.sum(dim=-1, keepdim=True)
        masked_attn = masked_attn / (attn_sum + 1e-8)      # [heads, seq]

        # Weighted sum: [heads, 1, seq] @ [heads, seq, hd] → [heads, hd]
        masked_output = torch.bmm(
            masked_attn.unsqueeze(1), values
        ).squeeze(1)

        return masked_output

    # ----- storage management -----

    def clear_claim_storage(self):
        """Clear all CLAIM-specific storage between forward passes."""
        self.attn_weights_storage.clear()
        self.value_storage.clear()
        self.oproj_input_storage.clear()
        self.storage.clear()


class CLAIMIntervenor(SteerManager):
    """
    Apply CLAIM shift vectors during inference by modifying o_proj inputs.

    Registers ``forward_pre_hook`` on ``o_proj`` linear layers so that
    the per-head attention output is shifted before the output projection:

    .. math::

        \\hat{O}_h^l = O_h^l + \\mathbb{I}_h^l \\, \\alpha \\, S_h^l

    Args:
        model: The VLM model.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        layer_name_template: Format string for self_attn layer paths.
    """

    def __init__(
        self,
        model,
        num_heads: int,
        head_dim: int,
        layer_name_template: str = "model.language_model.layers.{}.self_attn",
    ):
        super().__init__(model)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_name_template = layer_name_template
        self.shift_vectors: Dict[int, torch.Tensor] = {}
        self.target_heads: Set[Tuple[int, int]] = set()
        self.alpha: float = 1.0

    def set_intervention(
        self,
        shift_vectors: Dict[int, torch.Tensor],
        target_heads: Set[Tuple[int, int]],
        alpha: float = 1.0,
    ):
        """
        Configure the intervention parameters.

        Args:
            shift_vectors: Mapping ``layer_idx → tensor [num_heads, head_dim]``.
            target_heads: Set of ``(layer_idx, head_idx)`` pairs to intervene on.
            alpha: Intervention intensity scaling factor.
        """
        self.shift_vectors = shift_vectors
        self.target_heads = target_heads
        self.alpha = alpha

    def _intervention_pre_hook(self, layer_idx: int, module, args):
        """Modify o_proj input by adding shift vectors to identified heads."""
        inp = args[0]  # [batch, seq, num_heads * head_dim]

        shift = self.shift_vectors.get(layer_idx)
        if shift is None:
            return args

        shift = shift.to(device=inp.device, dtype=inp.dtype)  # [num_heads, head_dim]

        # Build per-head indicator mask
        head_mask = torch.zeros(self.num_heads, device=inp.device, dtype=inp.dtype)
        for (l, h) in self.target_heads:
            if l == layer_idx:
                head_mask[h] = 1.0

        if head_mask.sum() == 0:
            return args

        batch, seq_len, _ = inp.shape
        inp_reshaped = inp.view(batch, seq_len, self.num_heads, self.head_dim)
        # scaled_shift: [num_heads, head_dim], broadcast over batch and seq
        scaled_shift = self.alpha * shift * head_mask.unsqueeze(-1)
        modified = inp_reshaped + scaled_shift.unsqueeze(0).unsqueeze(0)
        modified = modified.reshape(batch, seq_len, -1)

        return (modified,)

    def attach_intervention(self, layer_indices: List[int]):
        """
        Attach intervention pre-hooks to ``o_proj`` at the given layers.

        Args:
            layer_indices: Decoder layer indices to intervene on.
        """
        for idx in layer_indices:
            o_name = f"{self.layer_name_template.format(idx)}.o_proj"
            o_layer = self._get_layer(o_name)
            hook = o_layer.register_forward_pre_hook(
                partial(self._intervention_pre_hook, idx)
            )
            self.hooks.append(hook)
        return self
