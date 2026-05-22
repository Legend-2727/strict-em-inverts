"""
Activation extractor for extracting hidden state activations from VLM models.
Supports layer-specific extraction with configurable token selection strategies.
"""

import torch
from typing import List, Optional, Union, Literal, Callable
import logging

logger = logging.getLogger(__name__)


class ActivationExtractor:
    """
    Extracts activations from specific layers of a Vision-Language Model.

    Uses forward hooks to capture hidden states during model inference.
    Supports extracting activations from the last token, first token, 
    or all tokens in the sequence.

    Example:
        >>> from src.models.smolvlm_wrapper import SmolVLMWrapper
        >>> wrapper = SmolVLMWrapper(quantization_4bit=True)
        >>> # Find layer path from model.named_modules()
        >>> for name, _ in wrapper.model.named_modules():
        ...     print(name)
        >>> extractor = ActivationExtractor(
        ...     wrapper.model, target_module="language_model.model.layers.8"
        ... )
        >>> extractor.register()
        >>> # Run inference...
        >>> activations = extractor.get_activations()
        >>> extractor.remove()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_module: str,
        token_position: Literal["last", "first", "all", "mean"] = "last",
    ):
        """
        Initialize the activation extractor.

        Args:
            model: The model to extract activations from (e.g., wrapper.model)
            target_module: String path to the target layer (e.g., "model.text_model.layers.15").
                Use `model.named_modules()` to find the correct path.
            token_position: Which token's activation to extract:
                - "last": Last token in sequence (default, good for causal LMs)
                - "first": First token in sequence
                - "all": All tokens (returns full sequence)
                - "mean": Mean over all tokens
        
        Example:
            >>> # Find layer path from model.named_modules()
            >>> for name, _ in model.named_modules():
            ...     print(name)
            >>> # Then use the path directly
            >>> extractor = ActivationExtractor(
            ...     model, target_module="model.text_model.layers.12"
            ... )
        """
        self.model = model
        self.target_module = target_module
        self.token_position = token_position

        self.activations: List[torch.Tensor] = []
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._target_layer: Optional[torch.nn.Module] = None

    def _get_layer_by_name(self, module_path: str) -> torch.nn.Module:
        """
        Get a layer by its string path using PyTorch's get_submodule.

        This is safer and more flexible than hardcoding attribute access.

        Args:
            module_path: Dot-separated path to the module (e.g., "model.text_model.layers.15")

        Returns:
            The target layer module

        Raises:
            ValueError: If the module path is not found in the model
        """
        try:
            return self.model.get_submodule(module_path)
        except AttributeError:
            raise ValueError(
                f"Could not find module '{module_path}' in the model. "
                "Use `for name, _ in model.named_modules(): print(name)` to find valid paths."
            )

    def _get_target_layer(self) -> torch.nn.Module:
        """
        Find and return the target layer using the specified module path.

        Returns:
            The target layer module

        Raises:
            ValueError: If the module path is not found
        """
        if self._target_layer is not None:
            return self._target_layer

        self._target_layer = self._get_layer_by_name(self.target_module)
        logger.debug(f"Using target module: {self.target_module}")
        return self._target_layer

    def _hook_fn(self, module: torch.nn.Module, input: tuple, output: tuple) -> None:
        """
        Hook function called during forward pass.

        Captures hidden states and applies token selection strategy.

        Args:
            module: The hooked module
            input: Input to the module
            output: Output from the module (usually tuple with hidden_states first)
        """
        # Extract hidden states from output
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Apply token position selection
        # Shape: [batch_size, seq_len, hidden_dim]
        if self.token_position == "last":
            activation = hidden_states[:, -1, :].detach().cpu()
        elif self.token_position == "first":
            activation = hidden_states[:, 0, :].detach().cpu()
        elif self.token_position == "mean":
            activation = hidden_states.mean(dim=1).detach().cpu()
        else:  # "all"
            activation = hidden_states.detach().cpu()

        self.activations.append(activation)

    def register(self) -> "ActivationExtractor":
        """
        Register the forward hook on the target layer.

        Returns:
            Self for method chaining
        """
        if self.hook_handle is not None:
            logger.warning("Hook already registered. Removing old hook first.")
            self.remove()

        layer = self._get_target_layer()
        self.hook_handle = layer.register_forward_hook(self._hook_fn)
        logger.info(
            f"Hook registered on {self.target_module} ({layer.__class__.__name__})")
        return self

    def remove(self) -> None:
        """Remove the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            logger.info(f"Hook removed from {self.target_module}")

    def clear(self) -> None:
        """Clear cached activations."""
        self.activations = []

    def get_activations(self, stack: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Get collected activations.

        Args:
            stack: If True, concatenate all activations along batch dimension.
                   If False, return list of individual activations.

        Returns:
            Stacked tensor or list of tensors
        """
        if not self.activations:
            logger.warning(
                "No activations collected. Run model inference first.")
            return torch.tensor([]) if stack else []

        if stack:
            return torch.cat(self.activations, dim=0)
        return self.activations

    @property
    def num_samples(self) -> int:
        """Number of activation samples collected."""
        return len(self.activations)

    def __enter__(self) -> "ActivationExtractor":
        """Context manager entry - register hook."""
        return self.register()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - remove hook."""
        self.remove()


class MultiLayerActivationExtractor:
    """
    Extracts activations from multiple layers simultaneously.

    Useful for analyzing how representations evolve across layers.

    Example:
        >>> extractor = MultiLayerActivationExtractor(
        ...     model,
        ...     target_modules=[
        ...         "model.text_model.layers.0",
        ...         "model.text_model.layers.8",
        ...         "model.text_model.layers.15",
        ...     ]
        ... )
        >>> with extractor:
        ...     # Run inference...
        ...     pass
        >>> activations = extractor.get_activations()
        >>> # activations is a dict: {module_path: tensor}
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_modules: List[str],
        token_position: Literal["last", "first", "all", "mean"] = "last"
    ):
        """
        Initialize multi-layer extractor.

        Args:
            model: The model to extract from
            target_modules: List of string paths to target layers.
                Use `model.named_modules()` to find the correct paths.
            token_position: Token selection strategy (applied to all layers)
        """
        self.model = model
        self.target_modules = target_modules
        self.token_position = token_position

        self.extractors = {
            module: ActivationExtractor(
                model, target_module=module, token_position=token_position)
            for module in target_modules
        }

    def register(self) -> "MultiLayerActivationExtractor":
        """Register hooks on all target layers."""
        for extractor in self.extractors.values():
            extractor.register()
        return self

    def remove(self) -> None:
        """Remove all hooks."""
        for extractor in self.extractors.values():
            extractor.remove()

    def clear(self) -> None:
        """Clear all cached activations."""
        for extractor in self.extractors.values():
            extractor.clear()

    def get_activations(self, stack: bool = True) -> dict:
        """
        Get activations from all layers.

        Args:
            stack: Whether to stack activations per layer

        Returns:
            Dictionary mapping module_path -> activations
        """
        return {
            idx: extractor.get_activations(stack=stack)
            for idx, extractor in self.extractors.items()
        }

    def __enter__(self) -> "MultiLayerActivationExtractor":
        return self.register()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()
