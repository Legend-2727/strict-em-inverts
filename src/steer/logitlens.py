"""
LogitLens: Decode intermediate hidden states into vocabulary space.

Implements the logit lens technique for vision-language models (specifically
Qwen2.5-VL). By projecting intermediate hidden states through the model's
final layer norm and unembedding matrix (lm_head), we can interpret what
the model "thinks" at each layer.

Mathematical foundation:
    p(V | v_i^l) = softmax(W_V · LayerNorm(v_i^l))

where v_i^l is the hidden state of token i at layer l, W_V is the
unembedding matrix, and LayerNorm is the final RMS norm.

See Also:
    - ``doc/logitlens.md`` for background.
    - ``src.steer.attentionviz`` for attention visualization patterns.
    - ``src.steer.steermanager`` for the base hook manager.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

from .steermanager import SteerManager


class LogitLensManager(SteerManager):
    """
    Manager for extracting hidden states and decoding them via the logit lens.

    Hooks into transformer decoder layers to capture intermediate hidden
    states, then projects them through the final layer norm and lm_head
    to produce vocabulary-space probabilities.

    Args:
        model: The VLM model (e.g. ``Qwen2_5_VLForConditionalGeneration``).
        tokenizer: The tokenizer / processor for decoding token IDs to strings.
        norm_layer_name: Dotted path to the final RMS norm layer.
        lm_head_name: Dotted path to the unembedding linear layer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        norm_layer_name: str = "model.language_model.norm",
        lm_head_name: str = "lm_head",
    ):
        super().__init__(model)
        self.tokenizer = tokenizer
        self.norm_layer = self._get_layer(norm_layer_name)
        self.lm_head = self._get_layer(lm_head_name)

    # ------------------------------------------------------------------
    # Probe helpers
    # ------------------------------------------------------------------

    def attach_decoder_probes(
        self,
        layer_indices: List[int],
        layer_name_template: str = "model.language_model.layers.{}",
    ) -> "LogitLensManager":
        """
        Attach extraction hooks to the specified decoder layers.

        Each hook captures the full sequence hidden states (``token_pos="all"``).

        Args:
            layer_indices: Decoder layer indices to probe.
            layer_name_template: Format string for layer paths.

        Returns:
            self (for chaining).
        """
        self._layer_names: List[str] = []
        for idx in layer_indices:
            name = layer_name_template.format(idx)
            self.attach_probe(name, token_pos="all")
            self._layer_names.append(name)
        return self

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_hidden_state(
        self, hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Project a hidden-state tensor into vocabulary logits.

        Applies the final layer norm followed by the lm_head linear layer.

        Args:
            hidden_state: Tensor of shape ``[seq_len, hidden_dim]`` or
                ``[batch, seq_len, hidden_dim]``.

        Returns:
            Logits tensor of the same leading dimensions with last dim
            equal to the vocabulary size.
        """
        device = next(self.lm_head.parameters()).device
        dtype = next(self.lm_head.parameters()).dtype
        h = hidden_state.to(device=device, dtype=dtype)
        h = self.norm_layer(h)
        return self.lm_head(h).float()  # float32 for numerical stability

    def decode_all_layers(
        self,
        top_k: int = 10,
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Decode captured hidden states for every probed layer and token.

        Returns:
            Nested dictionary::

                {
                    layer_name: {
                        token_index: {
                            "top_k_tokens": [("token_str", prob), ...],
                        },
                        ...
                    },
                    ...
                }
        """
        results: Dict[str, Dict[int, Dict[str, Any]]] = {}

        for layer_name in self._layer_names:
            acts_list = self.storage.get(layer_name, [])
            if not acts_list:
                continue

            # Take first forward pass (prefill), batch 0
            hidden = acts_list[0]  # [batch, seq_len, hidden_dim]
            if hidden.dim() == 3:
                hidden = hidden[0]  # [seq_len, hidden_dim]

            logits = self._decode_hidden_state(hidden)  # [seq_len, vocab]
            probs = F.softmax(logits, dim=-1)  # [seq_len, vocab]

            layer_result: Dict[int, Dict[str, Any]] = {}
            seq_len = probs.shape[0]

            for tok_idx in range(seq_len):
                token_probs = probs[tok_idx]  # [vocab]
                top_probs, top_ids = torch.topk(token_probs, top_k)

                top_k_tokens: List[Tuple[str, float]] = []
                for prob_val, tok_id in zip(top_probs.tolist(), top_ids.tolist()):
                    token_str = self.tokenizer.decode(
                        [tok_id], skip_special_tokens=False
                    )
                    top_k_tokens.append((token_str, prob_val))

                layer_result[tok_idx] = {
                    "top_k_tokens": top_k_tokens,
                }

            results[layer_name] = layer_result

        return results

    def decode_layer(
        self,
        layer_name: str,
        top_k: int = 10,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Decode a single layer's hidden states into vocabulary probabilities.

        Args:
            layer_name: The probed layer identifier.
            top_k: Number of top tokens to return per position.

        Returns:
            Dictionary mapping token index to its decoded information.
        """
        acts_list = self.storage.get(layer_name, [])
        if not acts_list:
            return {}

        hidden = acts_list[0]
        if hidden.dim() == 3:
            hidden = hidden[0]

        logits = self._decode_hidden_state(hidden)
        probs = F.softmax(logits, dim=-1)

        result: Dict[int, Dict[str, Any]] = {}
        for tok_idx in range(probs.shape[0]):
            token_probs = probs[tok_idx]
            top_probs, top_ids = torch.topk(token_probs, top_k)

            top_k_tokens = [
                (self.tokenizer.decode([tid], skip_special_tokens=False), p)
                for p, tid in zip(top_probs.tolist(), top_ids.tolist())
            ]
            result[tok_idx] = {
                "top_k_tokens": top_k_tokens,
            }
        return result

    # TODO: untested function - check later
    def decode_full_generation(
        self,
        layer_name: str,
        top_k: int = 10,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Decode a single layer's hidden states across the entire generation phase.
        
        Handles both the prefill phase (first forward pass) and the auto-regressive 
        generation steps, accounting for whether KV caching is enabled.

        Args:
            layer_name: The probed layer identifier.
            top_k: Number of top tokens to return per position.

        Returns:
            Dictionary mapping the absolute token index across the entire sequence 
            to its decoded information, including a 'phase' flag.
        """
        acts_list = self.storage.get(layer_name, [])
        if not acts_list:
            return {}

        all_hiddens = []
        
        for i, hidden in enumerate(acts_list):
            # Remove batch dimension if present [batch, seq_len, hidden_dim] -> [seq_len, hidden_dim]
            if hidden.dim() == 3:
                hidden = hidden[0] 

            if i == 0:
                # Prefill phase: keep the entire sequence
                all_hiddens.append(hidden)
            else:
                # Generation phase: 
                # If KV caching is ON, seq_len is 1. 
                # If KV caching is OFF, seq_len grows with each step, so we only want the last token.
                if hidden.shape[0] > 1:
                    hidden = hidden[-1:] # Slice to keep dimensions [1, hidden_dim]
                all_hiddens.append(hidden)

        # Concatenate all steps into a single continuous sequence [total_seq_len, hidden_dim]
        full_sequence_hidden = torch.cat(all_hiddens, dim=0)

        # Utilize the existing decoding projection
        logits = self._decode_hidden_state(full_sequence_hidden)
        probs = F.softmax(logits, dim=-1)

        result: Dict[int, Dict[str, Any]] = {}
        prefill_len = all_hiddens[0].shape[0]

        for tok_idx in range(probs.shape[0]):
            token_probs = probs[tok_idx]
            top_probs, top_ids = torch.topk(token_probs, top_k)

            top_k_tokens = [
                (self.tokenizer.decode([tid], skip_special_tokens=False), p)
                for p, tid in zip(top_probs.tolist(), top_ids.tolist())
            ]
            
            result[tok_idx] = {
                "top_k_tokens": top_k_tokens,
                "phase": "prefill" if tok_idx < prefill_len else "generation"
            }
            
        return result

    # ------------------------------------------------------------------
    # Vision-token visualization
    # ------------------------------------------------------------------

    def get_vision_token_top_predictions(
        self,
        layer_name: str,
        input_ids: torch.Tensor,
        image_token_id: int = 151655,
        top_k: int = 5,
    ) -> List[List[Tuple[str, float]]]:
        """
        Get top-k predicted tokens for each vision token position in a layer.

        Args:
            layer_name: The probed layer name.
            input_ids: Input IDs tensor ``[batch, seq_len]``.
            image_token_id: Token ID used for image placeholder tokens.
            top_k: Number of top tokens.

        Returns:
            List (one entry per vision token) of ``[(token_str, prob), ...]``.
        """
        acts_list = self.storage.get(layer_name, [])
        if not acts_list:
            return []

        hidden = acts_list[0]
        if hidden.dim() == 3:
            hidden = hidden[0]  # [seq_len, hidden_dim]

        vision_mask = (input_ids[0] == image_token_id)
        vision_hidden = hidden[vision_mask.cpu()]  # [n_vision, hidden_dim]

        logits = self._decode_hidden_state(vision_hidden)
        probs = F.softmax(logits, dim=-1)

        results: List[List[Tuple[str, float]]] = []
        for i in range(probs.shape[0]):
            top_probs, top_ids = torch.topk(probs[i], top_k)
            tokens = [
                (self.tokenizer.decode([tid], skip_special_tokens=False), p)
                for p, tid in zip(top_probs.tolist(), top_ids.tolist())
            ]
            results.append(tokens)
        return results


def save_logitlens_vision_overlay(
    top_predictions: List[List[Tuple[str, float]]],
    image_grid_thw: torch.Tensor,
    original_image: Image.Image,
    save_path: str,
    layer_name: str,
    alpha: float = 0.6,
    rank: int = 0,
    top_k: int = 3,
):
    """
    Overlay the top-k logit-lens predictions for each vision token onto the image.

    Creates a 2-D grid of text labels (top-k predicted tokens and their probabilities)
    placed at each vision token's spatial position, overlaid on the original image.

    Args:
        top_predictions: Output of
            ``LogitLensManager.get_vision_token_top_predictions``.
        image_grid_thw: ``[batch, 3]`` tensor with (T, H, W) grid dimensions.
        original_image: The original PIL image for the background.
        save_path: File path to save the figure.
        layer_name: Layer identifier (used in the title).
        alpha: Background image transparency when overlaid.
        rank: Which rank's probability to use for the heatmap coloring (0 = top-1).
        top_k: Number of top predictions to display in the text overlay.
    """
    grid_t, grid_h, grid_w = image_grid_thw[0].tolist()
    h_tokens = int(grid_h) // 2
    w_tokens = int(grid_w) // 2

    expected = h_tokens * w_tokens
    if len(top_predictions) != expected:
        raise ValueError(
            f"Token count mismatch: got {len(top_predictions)}, "
            f"expected {expected} ({h_tokens}x{w_tokens})"
        )

    # Build 2-D grids of labels and confidence values
    labels = []
    confidences = np.zeros((h_tokens, w_tokens))
    for r in range(h_tokens):
        row_labels = []
        for c in range(w_tokens):
            idx = r * w_tokens + c
            
            # Extract up to top_k predictions for the text label
            cell_texts = []
            max_k = min(top_k, len(top_predictions[idx]))
            for k in range(max_k):
                token_str, prob = top_predictions[idx][k]
                cell_texts.append(f"{token_str}: {prob:.2f}")
            
            # Join the top_k labels with newlines
            row_labels.append("\n".join(cell_texts))
            
            # Extract the probability for the heatmap color based on 'rank'
            # Default rank=0 uses the top-1 prediction's confidence
            _, rank_prob = top_predictions[idx][rank]
            confidences[r, c] = rank_prob
            
        labels.append(row_labels)

    fig, ax = plt.subplots(figsize=(max(10, w_tokens), max(10, h_tokens)))
    ax.imshow(original_image, extent=[0, w_tokens, h_tokens, 0], aspect="auto")
    ax.imshow(
        confidences,
        cmap="YlOrRd",
        alpha=alpha,
        extent=[0, w_tokens, h_tokens, 0],
    )

    # Place text labels at each grid cell center
    # Slightly reduced max fontsize (from 8 to 7) to better accommodate multiline text
    fontsize = max(3, min(7, 80 // max(h_tokens, w_tokens)))
    for r in range(h_tokens):
        for c in range(w_tokens):
            ax.text(
                c + 0.5, r + 0.5,
                labels[r][c],
                ha="center", va="center",
                fontsize=fontsize,
                color="black",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    facecolor="white",
                    alpha=0.6,
                    edgecolor="none",
                ),
            )

    ax.set_xlim(0, w_tokens)
    ax.set_ylim(h_tokens, 0)
    ax.axis("off")
    # Added Top K indicator to the plot title
    ax.set_title(
        f"Logit Lens: {layer_name}\nGrid: {w_tokens}x{h_tokens} (Top {top_k})",
        fontsize=12,
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_logitlens_confidence_heatmap(
    top_predictions: List[List[Tuple[str, float]]],
    image_grid_thw: torch.Tensor,
    original_image: Image.Image,
    save_path: str,
    layer_name: str,
    alpha: float = 0.5,
    rank: int = 0,
):
    """
    Save a heatmap of LogitLens top-1 confidence overlaid on the original image.

    Similar to the attention heatmap in ``attentionviz``, but the values
    represent the probability of the most-likely token at each vision
    position instead of attention weights.

    Args:
        top_predictions: Output of
            ``LogitLensManager.get_vision_token_top_predictions``.
        image_grid_thw: ``[batch, 3]`` tensor with (T, H, W).
        original_image: The original PIL image.
        save_path: Output path.
        layer_name: Layer identifier (used in title).
        alpha: Overlay opacity.
        rank: Top-k rank to plot (0 = top-1).
    """
    grid_t, grid_h, grid_w = image_grid_thw[0].tolist()
    h_tokens = int(grid_h) // 2
    w_tokens = int(grid_w) // 2

    expected = h_tokens * w_tokens
    if len(top_predictions) != expected:
        raise ValueError(
            f"Token count mismatch: got {len(top_predictions)}, "
            f"expected {expected} ({h_tokens}x{w_tokens})"
        )

    heatmap = np.zeros((h_tokens, w_tokens))
    for r in range(h_tokens):
        for c in range(w_tokens):
            idx = r * w_tokens + c
            _, prob = top_predictions[idx][rank]
            heatmap[r, c] = prob

    # Normalize for contrast
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(original_image)
    ax.imshow(
        heatmap,
        cmap="jet",
        alpha=alpha,
        extent=[0, original_image.width, original_image.height, 0],
    )
    ax.axis("off")
    ax.set_title(
        f"Logit Lens Confidence: {layer_name}\nGrid: {w_tokens}x{h_tokens}",
        fontsize=12,
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
