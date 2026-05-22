import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class Qwen3_4BWrapper:
    """
    Wrapper for Qwen3-4B-Instruct-2507 model.
    Non-thinking instruct model only (no <think> blocks).
    Uses BitsAndBytes 4-bit quantization by default.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
                 quantization_4bit: bool = True,
                 use_flash_attention: bool = True,
                 device_map: str = "auto",
                 quantization: Optional[str] = None):
        """
        Initialize the Qwen3-4B-Instruct-2507 wrapper.
        
        Args:
            model_name: HuggingFace model identifier for Qwen3-4B-Instruct-2507
            quantization_4bit: Whether to use 4-bit quantization (default: True).
                Deprecated in favor of `quantization`. Ignored when `quantization` is set.
            use_flash_attention: Whether to use flash attention 2 (recommended for speed/memory)
            device_map: Device mapping strategy
            quantization: Quantization mode: "nf4" (4-bit NormalFloat), "8bit", or "none".
                When set, overrides `quantization_4bit`. Default: None (falls back to
                quantization_4bit).
        """
        self.model_name = model_name
        # Resolve quantization mode
        if quantization is not None:
            assert quantization in ("nf4", "8bit", "none"), (
                f"quantization must be 'nf4', '8bit', or 'none', got '{quantization}'"
            )
            self.quantization = quantization
        else:
            self.quantization = "nf4" if quantization_4bit else "none"
        self.quantization_4bit = self.quantization == "nf4"
        self.use_flash_attention = use_flash_attention
        self.device_map = device_map
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.tokenizer = None
        
        self._load_model()
        
    def _load_model(self):
        """Load the model and tokenizer with optional 4-bit quantization."""
        logger.info(f"Loading Qwen3-4B-Instruct-2507 model: {self.model_name}")
        
        # Check for custom cache directory from environment variable
        cache_dir = os.environ.get("QWEN3_CACHE")
        if cache_dir:
            logger.info(f"Using custom cache directory: {cache_dir}")
        
        # Load tokenizer
        tokenizer_kwargs = {}
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            **tokenizer_kwargs
        )
        
        # Configure quantization if enabled
        quantization_config = None
        if self.quantization == "nf4":
            logger.info("Configuring 4-bit NF4 quantization with BitsAndBytes")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        elif self.quantization == "8bit":
            logger.info("Configuring 8-bit quantization with BitsAndBytes")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            logger.info("No quantization (full precision)")
        
        # Configure model loading
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
        
        # Add quantization config if enabled
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        
        # Use flash attention if available and requested
        if self.use_flash_attention and self.device == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        logger.info(f"Model loaded successfully on device: {self.device}")
        logger.info(f"Quantization mode: {self.quantization}")
    
    def raw_generate(self, 
                     text: str,
                     max_new_tokens: int = 16384,
                     **kwargs) -> str:
        """
        Generate text response without applying a chat template.

        Args:
            text: Raw text prompt to use directly (no chat template applied)
            max_new_tokens: Maximum number of new tokens to generate (default: 16384)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                **kwargs
            )
        
        # Trim input tokens from output
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        
        # Decode full response
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text.strip()

    def generate(self, 
                 messages: List[Dict],
                 max_new_tokens: int = 16384,
                 **kwargs) -> str:
        """
        Generate text response using the proper message format.
        
        Messages should follow the standard chat format:
        [
            {
                "role": "user",
                "content": "Your question here."
            }
        ]
        
        Args:
            messages: List of message dictionaries with role and content
            max_new_tokens: Maximum number of new tokens to generate (default: 16384)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Apply chat template (no enable_thinking needed for instruct model)
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
        )
        
        # Prepare inputs
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                **kwargs
            )
        
        # Trim input tokens from output
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        
        # Decode full response
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text.strip()
    
    def chat(self, 
             user_message: str,
             system_prompt: Optional[str] = None,
             **generation_kwargs) -> str:
        """
        Chat interface for instruction-tuned model.
        
        Args:
            user_message: User's message/question
            system_prompt: Optional system prompt to guide behavior
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Model's response as a string
        """
        # Create messages in the proper format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_message})
        
        return self.generate(
            messages, 
            **generation_kwargs
        )

    def batch_chat(self,
                   messages_list: List[List[Dict]],
                   max_new_tokens: int = 16384,
                   **kwargs) -> List[str]:
        """
        Batched chat interface for generating multiple responses in a single
        forward pass.

        Args:
            messages_list: List of message lists, each in the standard chat format.
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Additional generation parameters.

        Returns:
            List of generated text responses.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        if not messages_list:
            return []

        # Fall back to sequential generation for single item
        if len(messages_list) == 1:
            return [self.generate(messages_list[0], max_new_tokens=max_new_tokens, **kwargs)]

        # Apply chat template to each message list
        texts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_list
        ]

        # Tokenize with left padding for batch generation
        original_padding_side = self.tokenizer.padding_side
        original_pad_token = self.tokenizer.pad_token

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )

        # Restore original tokenizer settings
        self.tokenizer.padding_side = original_padding_side
        self.tokenizer.pad_token = original_pad_token

        # Extract only new tokens for each sequence
        input_length = inputs.input_ids.shape[1]
        results = []
        for i in range(len(messages_list)):
            output_ids = generated_ids[i][input_length:].tolist()
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            results.append(output_text.strip())

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "Model not loaded"}
        
        info = {
            "model_name": self.model_name,
            "quantization_4bit": self.quantization_4bit,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "use_flash_attention": self.use_flash_attention,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "context_length": 262144,
            "attention_heads": 32,
            "kv_heads": 8,
            "layers": 36,
        }
        
        # Add memory usage if available
        if torch.cuda.is_available():
            info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        
        return info
    
    def __repr__(self):
        return f"Qwen3_4BWrapper(model={self.model_name}, quantization_4bit={self.quantization_4bit}, flash_attn={self.use_flash_attention})"
