"""
Visual Attention Ratio (VAR): Quantify vision-token influence on generated tokens.

The VAR metric measures how much attention each newly generated text token
pays to the input image tokens across all layers and heads of a
vision-language model.

Mathematical definition:
    VAR^{(l,h)}(y_k) = sum_{i=1}^{n} A_k^{(l,h)}(a_k, i)

where A_k^{(l,h)}(a_k, i) is the attention weight from generated token y_k
to image token v_i at layer l, head h.

See Also:
    - ``doc/var_metric.md`` for background.
    - ``src.steer.attentionviz`` for attention visualization.
    - ``src.steer.steermanager`` for the base hook manager.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

from scipy.stats import pearsonr, spearmanr
from scipy.special import rel_entr

from .steermanager import SteerManager


class VARManager(SteerManager):
    """
    Manager for computing the Visual Attention Ratio (VAR) metric.

    Hooks into ``self_attn`` layers to capture post-softmax attention weights
    during autoregressive generation, then computes VAR by summing the
    attention from each generated token to all image-token positions.

    Requires ``use_flash_attention=False`` and ``attn_implementation="eager"``
    on the model, as Flash Attention does not return attention weights.

    Args:
        model: The VLM model (e.g. ``Qwen2_5_VLForConditionalGeneration``).
        image_token_id: Token ID for image placeholder tokens
            (Qwen2.5-VL default: ``151655``).
    """

    def __init__(self, model: torch.nn.Module, image_token_id: int = 151655):
        super().__init__(model)
        self.image_token_id = image_token_id
        self.attn_storage: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self._layer_indices: List[int] = []

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _attn_hook(self, layer_idx: int, module, input, output):
        """Capture post-softmax attention weights from ``self_attn``.

        Returns a modified output with attention weights replaced by ``None``
        so that the ~2.8 GiB-per-layer GPU tensors are freed immediately
        instead of accumulating across all 28 layers (~79 GiB total).
        """
        if isinstance(output, tuple) and len(output) > 1:
            if output[1] is None:
                raise ValueError(
                    "Attention weights are None. Ensure `output_attentions=True` "
                    "is set and the attention implementation supports returning "
                    "weights (flash_attention_2 does not)."
                )
            attn = output[1]
            if attn.shape[2] == 1:
                # Generation step: small tensor [batch, heads, 1, kv_len] — keep
                self.attn_storage[layer_idx].append(
                    attn.detach().cpu().to(torch.float32)
                )
            else:
                # Prefill step: huge [batch, heads, seq, seq] — store placeholder
                # compute_var() skips index 0 (prefill) so this is never read.
                self.attn_storage[layer_idx].append(None)
            # Free GPU memory by stripping attention weights from the output.
            return (output[0], None) + output[2:] if len(output) > 2 else (output[0], None)

    # ------------------------------------------------------------------
    # Probe attachment
    # ------------------------------------------------------------------

    def attach_var_probes(
        self,
        layer_indices: List[int],
        layer_name_template: str = "model.language_model.layers.{}.self_attn",
    ) -> "VARManager":
        """
        Attach attention extraction hooks to the specified decoder layers.

        Also sets ``model.config.output_attentions = True`` so that the
        self-attention modules return their weight matrices.

        Args:
            layer_indices: Decoder layer indices to probe.
            layer_name_template: Format string for ``self_attn`` layer paths.

        Returns:
            self (for chaining).
        """
        self.model.config.output_attentions = True
        self._layer_indices = sorted(layer_indices)
        for idx in layer_indices:
            name = layer_name_template.format(idx)
            layer = self._get_layer(name)
            hook = layer.register_forward_hook(partial(self._attn_hook, idx))
            self.hooks.append(hook)
        return self

    # ------------------------------------------------------------------
    # VAR computation
    # ------------------------------------------------------------------

    def compute_var(
        self,
        input_ids: torch.Tensor,
    ) -> Dict[int, np.ndarray]:
        """
        Compute VAR for all generated tokens across all probed layers and heads.

        During autoregressive generation the model performs:
          - Step 0 (prefill): processes the full input sequence.
          - Step k >= 1: generates one new token.

        For each generation step the attention tensor has shape
        ``[batch, heads, 1, kv_len]``. We sum the attention weights
        directed at image-token positions to obtain VAR.

        Args:
            input_ids: The **original** (pre-generation) input IDs tensor of
                shape ``[batch, seq_len]``.

        Returns:
            Dictionary mapping generated-token index (0-based) to a numpy
            array of shape ``[num_layers, num_heads]`` with VAR values.
        """
        # Identify image-token positions in the original input
        image_positions = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]

        if len(image_positions) == 0:
            raise ValueError(
                f"No image tokens (id={self.image_token_id}) found in input_ids."
            )

        # Number of forward-pass steps captured (prefill + generation tokens)
        any_layer = self._layer_indices[0]
        num_steps = len(self.attn_storage[any_layer])
        num_generated = num_steps - 1  # skip prefill

        if num_generated <= 0:
            return {}

        num_heads = self.attn_storage[any_layer][1].shape[1]  # index 1 = first gen step (index 0 is prefill placeholder)
        num_layers = len(self._layer_indices)

        var_results: Dict[int, np.ndarray] = {}

        for gen_idx in range(num_generated):
            step_idx = gen_idx + 1  # offset past prefill
            var_matrix = np.zeros((num_layers, num_heads), dtype=np.float32)

            for layer_pos, layer_idx in enumerate(self._layer_indices):
                attn = self.attn_storage[layer_idx][step_idx]
                # attn shape during generation: [batch, heads, 1, kv_len]
                attn_from_new_token = attn[0, :, -1, :]  # [heads, kv_len]
                attn_to_images = attn_from_new_token[:, image_positions]  # [heads, n_img]
                var_matrix[layer_pos] = attn_to_images.sum(dim=-1).numpy()

            var_results[gen_idx] = var_matrix

        return var_results

    # ------------------------------------------------------------------
    # Storage management
    # ------------------------------------------------------------------

    def clear_var_storage(self):
        """Clear all attention storage between runs."""
        self.attn_storage.clear()
        self.storage.clear()


# ======================================================================
# Heads Guided Attention Intervention
# ======================================================================

# Stores the original ``eager_attention_forward`` so the patched version
# can fall back to it for layers that are *not* targeted by the intervention.
_original_eager_attention_forward = None


def _intervened_eager_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Drop-in replacement for ``eager_attention_forward`` with optional
    per-layer visual-attention intervention.

    When a ``self_attn`` module carries an active ``_var_intervention``
    config dict the function enhances the pre-softmax attention scores
    directed at image-token positions before applying softmax.  Otherwise
    the original implementation is called unchanged.

    The enhancement follows the formula from ``doc/var_intervention.md``:

    .. math::

        \\hat{S}_k^{(l,h)}(a_k, i) = S_k^{(l,h)}(a_k, i)
            + \\alpha \\frac{1}{H} \\sum_{h=1}^{H} |S_k^{(l,h)}(a_k, i)|

    where *i* iterates over image-token positions.
    """
    cfg = getattr(module, "_var_intervention", None)

    # --- Fast path: no intervention configured for this module ----------
    if cfg is None or not cfg.get("active", False):
        return _original_eager_attention_forward(
            module, query, key, value, attention_mask,
            scaling, dropout, **kwargs,
        )

    # --- Intervention path ----------------------------------------------
    from transformers.models.qwen2_vl.modeling_qwen2_vl import repeat_kv

    image_positions = cfg["image_positions"]
    alpha = cfg["alpha"]

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # Pre-softmax attention scores  [batch, heads, q_len, kv_len]
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # --- Enhance attention to image tokens (before causal mask) ---------
    if image_positions is not None and len(image_positions) > 0:
        img_pos = image_positions.to(attn_weights.device)
        # Guard against positions beyond current kv_len (shouldn't happen)
        kv_len = attn_weights.shape[-1]
        img_pos = img_pos[img_pos < kv_len]

        if len(img_pos) > 0:
            # [batch, heads, q_len, n_img]
            img_scores = attn_weights[:, :, :, img_pos]
            # Mean of absolute scores across heads -> [batch, 1, q_len, n_img]
            enhancement = img_scores.abs().mean(dim=1, keepdim=True)
            attn_weights[:, :, :, img_pos] = img_scores + alpha * enhancement

    # Apply causal mask *after* the intervention
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # Standard softmax -> weighted sum
    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32,
    ).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(
        attn_weights, p=dropout, training=module.training,
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class VARInterventionManager(VARManager):
    """VARManager extended with the Heads Guided Attention Intervention.

    In addition to the standard VAR extraction probes this manager can
    *modify* the pre-softmax attention scores inside the targeted layer
    range ``[layer_start, layer_end]`` so that attention to image tokens
    is strengthened (see ``doc/var_intervention.md``).

    The intervention is toggled on/off per sample via
    :meth:`activate_intervention` / :meth:`deactivate_intervention` so
    that the same manager instance can be used for both baseline and
    intervened runs within a single experiment.

    Args:
        model: The VLM model (e.g. ``Qwen2_5_VLForConditionalGeneration``).
        image_token_id: Token ID for image placeholder tokens.
        layer_start: First layer index (inclusive) for the intervention.
        layer_end: Last layer index (inclusive) for the intervention.
        alpha: Balance factor controlling intervention strength.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        image_token_id: int = 151655,
        layer_start: Optional[int] = None,
        layer_end: Optional[int] = None,
        alpha: float = 0.5,
    ):
        super().__init__(model, image_token_id)
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.alpha = alpha
        self._intervention_active = False
        self._image_positions: Optional[torch.Tensor] = None
        self._patched = False
        self._layer_name_template = "model.language_model.layers.{}.self_attn"

    # ------------------------------------------------------------------
    # Global monkey-patch management
    # ------------------------------------------------------------------

    def _patch_attention(self) -> None:
        """Replace ``eager_attention_forward`` in the Qwen2-VL module."""
        global _original_eager_attention_forward

        if self._patched:
            return

        import transformers.models.qwen2_vl.modeling_qwen2_vl as _qwen_module

        _original_eager_attention_forward = _qwen_module.eager_attention_forward
        _qwen_module.eager_attention_forward = _intervened_eager_attention_forward
        self._patched = True

    def _unpatch_attention(self) -> None:
        """Restore the original ``eager_attention_forward``."""
        global _original_eager_attention_forward

        if not self._patched:
            return

        import transformers.models.qwen2_vl.modeling_qwen2_vl as _qwen_module

        _qwen_module.eager_attention_forward = _original_eager_attention_forward
        _original_eager_attention_forward = None
        self._patched = False

    # ------------------------------------------------------------------
    # Per-layer config helpers
    # ------------------------------------------------------------------

    def _set_layer_configs(self, active: bool) -> None:
        """Write ``_var_intervention`` config dicts on targeted ``self_attn`` modules."""
        if self.layer_start is None or self.layer_end is None:
            return
        for layer_idx in range(self.layer_start, self.layer_end + 1):
            name = self._layer_name_template.format(layer_idx)
            module = self._get_layer(name)
            module._var_intervention = {
                "active": active,
                "image_positions": self._image_positions,
                "alpha": self.alpha,
            }

    def _clear_layer_configs(self) -> None:
        """Remove ``_var_intervention`` attributes from targeted modules."""
        if self.layer_start is None or self.layer_end is None:
            return
        for layer_idx in range(self.layer_start, self.layer_end + 1):
            name = self._layer_name_template.format(layer_idx)
            module = self._get_layer(name)
            if hasattr(module, "_var_intervention"):
                del module._var_intervention

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image_positions(self, input_ids: torch.Tensor) -> None:
        """Detect image-token positions from *input_ids* (batch dim 0)."""
        self._image_positions = (
            (input_ids[0] == self.image_token_id)
            .nonzero(as_tuple=True)[0]
            .cpu()
        )

    def activate_intervention(
        self, input_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Activate the attention intervention for subsequent forward passes.

        Args:
            input_ids: If provided the image-token positions are recomputed.
        """
        if self.layer_start is None or self.layer_end is None:
            raise ValueError(
                "Cannot activate intervention: layer_start and layer_end "
                "must be set (not None)."
            )
        if input_ids is not None:
            self.set_image_positions(input_ids)
        self._patch_attention()
        self._set_layer_configs(active=True)
        self._intervention_active = True

    def deactivate_intervention(self) -> None:
        """Deactivate the intervention (layers revert to standard attention)."""
        self._set_layer_configs(active=False)
        self._intervention_active = False

    @property
    def intervention_active(self) -> bool:
        return self._intervention_active

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Remove all hooks, clear intervention configs, and unpatch."""
        self.deactivate_intervention()
        self._clear_layer_configs()
        self._unpatch_attention()
        super().remove_hooks()


# ======================================================================
# Visualization helpers
# ======================================================================


def save_var_heatmap(
    var_matrix: np.ndarray,
    token_text: str,
    token_idx: int,
    layer_indices: List[int],
    save_path: str,
    figsize: tuple = (12, 8),
    cmap: str = "viridis",
):
    """
    Save a VAR heatmap for a single generated token.

    X-axis = attention heads, Y-axis = layers, intensity = VAR value.

    Args:
        var_matrix: Array of shape ``[num_layers, num_heads]``.
        token_text: Decoded text of the generated token.
        token_idx: 0-based generation index.
        layer_indices: Layer indices (for Y-axis tick labels).
        save_path: Output image path.
        figsize: Figure size in inches.
        cmap: Matplotlib colour-map name.
    """
    num_layers, num_heads = var_matrix.shape

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(var_matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xlabel("Attention Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(
        f'VAR Heatmap \u2014 Token {token_idx}: "{token_text}"',
        fontsize=14,
    )

    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f"H{h}" for h in range(num_heads)], fontsize=8)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"L{l}" for l in layer_indices], fontsize=8)

    fig.colorbar(im, ax=ax, label="VAR (attention to image tokens)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_var_summary_heatmap(
    var_results: Dict[int, np.ndarray],
    generated_tokens: List[str],
    layer_indices: List[int],
    save_path: str,
    figsize: Optional[tuple] = None,
    cmap: str = "viridis",
):
    """
    Save a summary heatmap averaging VAR across heads for every generated token.

    X-axis = generated tokens, Y-axis = layers, intensity = mean VAR over heads.

    Args:
        var_results: Output of :meth:`VARManager.compute_var`.
        generated_tokens: List of decoded token strings.
        layer_indices: Layer indices (for Y-axis tick labels).
        save_path: Output image path.
        figsize: Figure size (auto-scaled if ``None``).
        cmap: Matplotlib colour-map name.
    """
    num_tokens = len(var_results)
    num_layers = len(layer_indices)

    if num_tokens == 0:
        return

    # Build [num_layers, num_tokens] matrix of head-averaged VAR
    summary = np.zeros((num_layers, num_tokens), dtype=np.float32)
    for tok_idx in range(num_tokens):
        if tok_idx in var_results:
            summary[:, tok_idx] = var_results[tok_idx].mean(axis=1)

    if figsize is None:
        figsize = (max(10, num_tokens * 0.6), max(8, num_layers * 0.3))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(summary, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xlabel("Generated Token", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title("VAR Summary (mean over heads)", fontsize=14)

    token_labels = [
        f"{i}: {generated_tokens[i]}" if i < len(generated_tokens) else str(i)
        for i in range(num_tokens)
    ]
    ax.set_xticks(range(num_tokens))
    ax.set_xticklabels(token_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"L{l}" for l in layer_indices], fontsize=8)

    fig.colorbar(im, ax=ax, label="Mean VAR")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ======================================================================
# Cross-lingual VAR comparison metrics
# ======================================================================

_EPS = 1e-12  # small constant to avoid division-by-zero / log(0)


def _normalise_to_distribution(mat: np.ndarray) -> np.ndarray:
    """Normalise a matrix so that all elements sum to 1 (probability dist)."""
    total = mat.sum()
    if total < _EPS:
        # uniform fallback when the matrix is all zeros
        return np.ones_like(mat) / mat.size
    return mat / total


def mean_difference(mat_a: np.ndarray, mat_b: np.ndarray) -> float:
    """
    Overall mean difference: mean(A) - mean(B).

    A positive value indicates that matrix A has higher average visual
    grounding than B.
    """
    return float(mat_a.mean() - mat_b.mean())


def variance_difference(mat_a: np.ndarray, mat_b: np.ndarray) -> float:
    """
    Difference of variances: var(A) - var(B).

    A positive value means A's attention is more spread out (higher variance)
    than B's.
    """
    return float(mat_a.var() - mat_b.var())


def mse(mat_a: np.ndarray, mat_b: np.ndarray) -> float:
    """
    Mean Squared Error between two VAR matrices.

    Lower values indicate more similar attention patterns.
    """
    return float(np.mean((mat_a - mat_b) ** 2))


def pearson_correlation(mat_a: np.ndarray, mat_b: np.ndarray) -> Tuple[float, float]:
    """
    Pearson correlation between flattened VAR matrices.

    Returns:
        (correlation_coefficient, p_value).
        ~1.0 means identical grounding structure.
    """
    r, p = pearsonr(mat_a.flatten(), mat_b.flatten())
    return float(r), float(p)


def kl_divergence(mat_a: np.ndarray, mat_b: np.ndarray) -> float:
    """
    KL divergence KL(A || B) after normalising both matrices to probability
    distributions.

    Information-theoretic distance between the two attention allocations.
    """
    p = _normalise_to_distribution(mat_a).flatten() + _EPS
    q = _normalise_to_distribution(mat_b).flatten() + _EPS
    # re-normalise after adding epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(rel_entr(p, q).sum())


def attention_entropy(mat: np.ndarray) -> float:
    """
    Shannon entropy of the normalised VAR matrix.

    High entropy → diffuse attention across heads/layers.
    Low entropy  → concentrated on specific heads/layers.
    """
    p = _normalise_to_distribution(mat).flatten()
    p = p[p > 0]  # drop zeros
    return float(-np.sum(p * np.log(p)))


def gini_coefficient(mat: np.ndarray) -> float:
    """
    Gini coefficient of the VAR matrix values.

    Measures inequality of attention allocation.
    0 → perfectly equal, 1 → maximally concentrated.
    """
    vals = np.sort(mat.flatten())
    n = vals.size
    if n == 0 or vals.sum() < _EPS:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * vals) / (n * vals.sum())) - (n + 1) / n)


def headwise_correlation(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Pearson correlation for each corresponding head across layers.

    For head *h*, correlate ``mat_a[:, h]`` with ``mat_b[:, h]``.

    Args:
        mat_a: VAR matrix of shape ``[num_layers, num_heads]``.
        mat_b: VAR matrix of shape ``[num_layers, num_heads]``.

    Returns:
        Dictionary with keys:
        - ``"correlations"``: 1-D array of length ``num_heads``
        - ``"p_values"``:     1-D array of length ``num_heads``
        - ``"mean_correlation"``: scalar mean over heads
    """
    num_heads = mat_a.shape[1]
    corrs = np.zeros(num_heads, dtype=np.float64)
    pvals = np.zeros(num_heads, dtype=np.float64)
    for h in range(num_heads):
        col_a = mat_a[:, h]
        col_b = mat_b[:, h]
        # If either column is constant, correlation is undefined → 0
        if col_a.std() < _EPS or col_b.std() < _EPS:
            corrs[h] = 0.0
            pvals[h] = 1.0
        else:
            r, p = pearsonr(col_a, col_b)
            corrs[h] = r
            pvals[h] = p
    return {
        "correlations": corrs,
        "p_values": pvals,
        "mean_correlation": float(corrs.mean()),
    }


def linear_cka(mat_a: np.ndarray, mat_b: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment (CKA) between two VAR matrices.

    Treats each matrix as a representation where rows = layers (examples) and
    columns = heads (features).  CKA measures structural similarity
    independent of isotropic scaling.

    CKA ~ 1.0 → nearly identical structure.
    CKA ~ 0.0 → unrelated.

    Reference:
        Kornblith et al., "Similarity of Neural Network Representations
        Revisited", ICML 2019.
    """
    X = mat_a.astype(np.float64)
    Y = mat_b.astype(np.float64)

    # Centre columns
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC with linear kernel: HSIC(X,Y) = ||Y^T X||_F^2 / (n-1)^2
    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    YtX = Y.T @ X  # [heads, heads]
    XtX = X.T @ X
    YtY = Y.T @ Y

    hsic_xy = np.sum(YtX ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < _EPS:
        return 0.0
    return float(hsic_xy / denom)


def compute_all_var_comparison_metrics(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
) -> Dict:
    """
    Compute all cross-lingual VAR comparison metrics between two matrices.

    Args:
        mat_a: VAR matrix for language A, shape ``[num_layers, num_heads]``.
        mat_b: VAR matrix for language B, shape ``[num_layers, num_heads]``.
        label_a: Label for language A (e.g. ``"hindi"``).
        label_b: Label for language B (e.g. ``"english"``).

    Returns:
        Dictionary of metric names to values.
    """
    pearson_r, pearson_p = pearson_correlation(mat_a, mat_b)
    hw = headwise_correlation(mat_a, mat_b)

    return {
        "mean_difference": mean_difference(mat_a, mat_b),
        "variance_difference": variance_difference(mat_a, mat_b),
        "mse": mse(mat_a, mat_b),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "kl_divergence": kl_divergence(mat_a, mat_b),
        f"entropy_{label_a}": attention_entropy(mat_a),
        f"entropy_{label_b}": attention_entropy(mat_b),
        "entropy_difference": attention_entropy(mat_a) - attention_entropy(mat_b),
        f"gini_{label_a}": gini_coefficient(mat_a),
        f"gini_{label_b}": gini_coefficient(mat_b),
        "gini_difference": gini_coefficient(mat_a) - gini_coefficient(mat_b),
        "headwise_mean_correlation": hw["mean_correlation"],
        "headwise_correlations": hw["correlations"].tolist(),
        "linear_cka": linear_cka(mat_a, mat_b),
    }
