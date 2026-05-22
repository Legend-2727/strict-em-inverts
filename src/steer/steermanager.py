"""
SteerManager: Unified activation extraction and steering for neural networks.

This module provides the `SteerManager` class for attaching PyTorch forward hooks
to neural network layers, enabling:
    - **Activation extraction**: Capture hidden states at specific layers for analysis.
    - **Activation steering**: Modify hidden states in-place to influence model behavior.

The manager supports multi-layer operations, automatic VRAM management (by moving
extracted activations to CPU), and clean resource management via context managers.

Typical usage:
    >>> from src.steer.steermanager import SteerManager
    >>> manager = SteerManager(model)
    >>> manager.attach_probe("model.layers.15")
    >>> output = model(**inputs)
    >>> activations = manager.get_activations("model.layers.15")
    >>> manager.remove_hooks()

See Also:
    - `src.activation` for contrastive vector computation
    - `src.steer` for steering vector utilities
"""

import torch
from collections import defaultdict
from typing import Literal, Callable, Optional, Union, List, Dict
from functools import partial


class SteerManager:
    """
    Unified manager for extracting and modifying model activations via PyTorch hooks.

    This class provides a clean interface for attaching forward hooks to model layers,
    supporting both read-only probes (for activation extraction) and modification hooks
    (for steering/intervention experiments). It handles HuggingFace-style tuple outputs
    automatically and manages hook lifecycle.

    Attributes:
        model (torch.nn.Module): The neural network model to attach hooks to.
        hooks (List[torch.utils.hooks.RemovableHandle]): Active hook handles.
        storage (Dict[str, List[torch.Tensor]]): Extracted activations keyed by layer name.

    Example:
        Basic activation extraction::

            manager = SteerManager(model)
            manager.attach_probe("model.layers.15", token_pos="last")

            with torch.no_grad():
                model(**inputs)

            activations = manager.get_activations("model.layers.15")
            manager.remove_hooks()

        Using as context manager::

            with SteerManager(model) as manager:
                manager.attach_probe("model.layers.15")
                model(**inputs)
                activations = manager.get_activations("model.layers.15")
            # Hooks automatically removed on exit

        Steering with a contrastive vector::

            steer_vector = positive_acts.mean(0) - negative_acts.mean(0)
            steer_fn = lambda x: x + 0.5 * steer_vector.to(x.device)

            manager = SteerManager(model)
            manager.attach_steering("model.layers.15", steer_fn)
            output = model.generate(**inputs)
            manager.remove_hooks()

    Note:
        - Extracted activations are moved to CPU to conserve GPU VRAM.
        - Multiple probes/steering hooks can be attached to different layers.
        - Call `remove_hooks()` when done to prevent memory leaks and side effects.
    """

    def __init__(self, model: torch.nn.Module):
        """
        Initialize the SteerManager with a target model.

        Args:
            model: The PyTorch model to attach hooks to. Can be any `torch.nn.Module`,
                including HuggingFace transformers.
        """
        self.model = model
        self.hooks = []
        # Storage: {layer_name: [tensor1, tensor2, ...]}
        self.storage: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def _get_layer(self, layer_name: str) -> torch.nn.Module:
        """
        Retrieve a submodule from the model by its dotted path name.

        Args:
            layer_name: Dotted path to the layer (e.g., "model.layers.15" or
                "model.vision_model.encoder.layers.0").

        Returns:
            The requested submodule.

        Raises:
            ValueError: If the layer path does not exist in the model.

        Example:
            >>> layer = manager._get_layer("model.layers.15")
            >>> print(type(layer))  # e.g., LlamaDecoderLayer
        """
        try:
            return self.model.get_submodule(layer_name)
        except AttributeError:
            raise ValueError(f"Layer '{layer_name}' not found in model.")

    def _extract_hook(self, layer_name: str, token_pos: str, module, input, output):
        """
        Forward hook callback that extracts and stores layer activations.

        This method is registered as a forward hook via `attach_probe()`. It handles
        HuggingFace-style tuple outputs (where hidden states are the first element)
        and applies token position selection before storing.

        Args:
            layer_name: Identifier for storage (passed via `functools.partial`).
            token_pos: Token selection strategy ("last", "first", "mean", or "all").
            module: The layer module (unused, required by hook signature).
            input: Layer input tensors (unused, required by hook signature).
            output: Layer output - either a tensor or tuple with hidden states first.

        Note:
            - Activations are detached and moved to CPU to conserve GPU memory.
            - Multiple forward passes accumulate in storage; call `clear_storage()` to reset.
        """
        # Handle HuggingFace tuple outputs
        hidden_states = output[0] if isinstance(output, tuple) else output

        # Token selection logic
        if token_pos == "last":
            act = hidden_states[:, -1, :]
        elif token_pos == "first":
            act = hidden_states[:, 0, :]
        elif token_pos == "mean":
            act = hidden_states.mean(dim=1)
        else:  # "all"
            act = hidden_states

        self.storage[layer_name].append(act.detach().cpu())

    def _modify_hook(self, modification_fn: Callable, module, input, output):
        """
        Forward hook callback that modifies layer activations for steering.

        This method is registered as a forward hook via `attach_steering()`. It applies
        a user-provided transformation function to the hidden states, enabling activation
        steering experiments (e.g., adding contrastive vectors).

        Args:
            modification_fn: A callable that takes a tensor of hidden states and returns
                the modified tensor. Must preserve tensor shape.
            module: The layer module (unused, required by hook signature).
            input: Layer input tensors (unused, required by hook signature).
            output: Layer output - either a tensor or tuple with hidden states first.

        Returns:
            Modified output in the same format as input (tensor or tuple).

        Example:
            A typical modification function for contrastive steering::

                steer_vec = (positive_mean - negative_mean).to(device)
                modify_fn = lambda x: x + alpha * steer_vec
        """
        # Handle tuple outputs (common in HF models)
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Apply function to hidden states
            modified = modification_fn(hidden_states)
            # Return reconstructed tuple
            return (modified,) + output[1:]
        else:
            return modification_fn(output)

    def attach_probe(
        self,
        layer_name: str,
        token_pos: Literal["last", "first", "all", "mean"] = "last"
    ):
        """
        Attach a read-only hook to extract activations from a layer.

        Registers a forward hook that captures hidden states during model inference.
        Activations are stored in `self.storage[layer_name]` and can be retrieved
        with `get_activations()`.

        Args:
            layer_name: Dotted path to the target layer (e.g., "model.layers.15").
            token_pos: Which token position(s) to extract:
                - "last": Final token only (default, useful for causal LMs).
                - "first": First token only (e.g., [CLS] token for BERT-style).
                - "mean": Mean-pooled across sequence dimension.
                - "all": Full sequence (shape: [batch, seq_len, hidden_dim]).

        Returns:
            self: Enables method chaining.

        Example:
            >>> manager.attach_probe("model.layers.15", token_pos="last")
            >>> manager.attach_probe("model.layers.20", token_pos="mean")
            >>> model(**inputs)  # Both layers now captured
        """
        layer = self._get_layer(layer_name)
        hook = layer.register_forward_hook(
            partial(self._extract_hook, layer_name, token_pos)
        )
        self.hooks.append(hook)
        return self

    def attach_steering(
        self,
        layer_name: str,
        steer_fn: Callable[[torch.Tensor], torch.Tensor]
    ):
        """
        Attach a modification hook to steer activations at a layer.

        Registers a forward hook that transforms hidden states in-place during
        model inference. This enables activation steering experiments such as
        adding contrastive vectors to influence model behavior.

        Args:
            layer_name: Dotted path to the target layer (e.g., "model.layers.15").
            steer_fn: A function that takes hidden states tensor [batch, seq, hidden_dim]
                and returns a modified tensor of the same shape.

        Returns:
            self: Enables method chaining.

        Example:
            Add a contrastive steering vector::

                steer_vec = positive_acts.mean(0) - negative_acts.mean(0)
                steer_fn = lambda x: x + 0.5 * steer_vec.to(x.device)
                manager.attach_steering("model.layers.15", steer_fn)

            Scale activations::

                manager.attach_steering("model.layers.10", lambda x: x * 1.2)

        Warning:
            Steering hooks modify model behavior. Always call `remove_hooks()`
            after experiments to restore normal model operation.
        """
        layer = self._get_layer(layer_name)
        hook = layer.register_forward_hook(
            partial(self._modify_hook, steer_fn)
        )
        self.hooks.append(hook)
        return self

    def get_activations(self, layer_name: str, stack: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Retrieve stored activations for a specific layer.

        Returns activations captured by probes attached to the specified layer.
        Multiple forward passes accumulate activations which can be stacked or
        returned as a list.

        Args:
            layer_name: The layer identifier used when attaching the probe.
            stack: If True (default), concatenate all batches along dim=0.
                If False, return as a list of tensors from each forward pass.

        Returns:
            If stack=True: A single tensor with all activations concatenated,
                or an empty tensor if no activations were captured.
            If stack=False: A list of tensors, one per forward pass,
                or an empty list if no activations were captured.

        Example:
            >>> manager.attach_probe("model.layers.15")
            >>> for batch in dataloader:
            ...     model(**batch)
            >>> all_acts = manager.get_activations("model.layers.15")  # Stacked
            >>> per_batch = manager.get_activations("model.layers.15", stack=False)
        """
        data = self.storage.get(layer_name, [])
        if not data:
            return torch.tensor([]) if stack else []
        return torch.cat(data, dim=0) if stack else data

    def clear_storage(self) -> None:
        """
        Clear all stored activations without removing hooks.

        Use this to reset storage between different data batches or experiments
        while keeping the same probes attached.

        Example:
            >>> manager.attach_probe("model.layers.15")
            >>> model(**positive_samples)
            >>> positive_acts = manager.get_activations("model.layers.15")
            >>> manager.clear_storage()
            >>> model(**negative_samples)
            >>> negative_acts = manager.get_activations("model.layers.15")
        """
        self.storage.clear()

    def remove_hooks(self) -> None:
        """
        Remove all registered hooks from the model.

        This should be called when done with extraction/steering to:
        - Prevent memory leaks from lingering hook references.
        - Restore normal model behavior (especially after steering).
        - Allow garbage collection of stored tensors.

        Note:
            This does NOT clear stored activations. Call `clear_storage()`
            separately if needed.

        Example:
            >>> manager.attach_probe("model.layers.15")
            >>> model(**inputs)
            >>> activations = manager.get_activations("model.layers.15")
            >>> manager.remove_hooks()  # Clean up when done
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self) -> "SteerManager":
        """
        Enter context manager: clears any existing storage.

        Returns:
            self: The SteerManager instance for use in the `with` block.
        """
        self.storage.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit context manager: removes all hooks automatically.

        This ensures hooks are cleaned up even if an exception occurs,
        preventing memory leaks and restoring normal model behavior.

        Args:
            exc_type: Exception type if an error occurred, else None.
            exc_val: Exception value if an error occurred, else None.
            exc_tb: Exception traceback if an error occurred, else None.
        """
        self.remove_hooks()
