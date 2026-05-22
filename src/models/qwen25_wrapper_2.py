import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers.image_utils import load_image
from PIL import Image
import logging
from typing import List, Union, Optional, Dict, Any

logger = logging.getLogger(__name__)


class Qwen25Wrapper2:
    """
    Wrapper for Qwen2.5-VL-3B-Instruct model.
    Uses BitsAndBytes 4-bit quantization by default.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 quantization_4bit: bool = True,
                 use_flash_attention: bool = True,
                 device_map: str = "auto",
                 min_pixels: Optional[int] = None,
                 max_pixels: Optional[int] = None,
                 quantization: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Qwen2.5-VL wrapper.
        
        Args:
            model_name: HuggingFace model identifier for Qwen2.5-VL
            quantization_4bit: Whether to use 4-bit quantization (default: True).
                Deprecated in favor of `quantization`. Ignored when `quantization` is set.
            use_flash_attention: Whether to use flash attention 2 (recommended for speed/memory)
            device_map: Device mapping strategy
            min_pixels: Minimum pixels for image resizing (default: 256*28*28)
            max_pixels: Maximum pixels for image resizing (default: 1280*28*28)
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
            # Backwards compatibility: map old bool flag
            self.quantization = "nf4" if quantization_4bit else "none"
        self.quantization_4bit = self.quantization == "nf4"  # keep for get_model_info etc.
        self.use_flash_attention = use_flash_attention
        self.device_map = device_map
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model_kwargs_extra = kwargs
        
        self.model = None
        self.processor = None
        
        self._load_model()
        
    def _load_model(self):
        """Load the model and processor with optional 4-bit quantization."""
        logger.info(f"Loading Qwen2.5-VL model: {self.model_name}")
        
        # Check for custom cache directory from environment variable
        cache_dir = os.environ.get("QWEN25VLINS_CACHE")
        if cache_dir:
            logger.info(f"Using custom cache directory: {cache_dir}")
        
        # Load processor with optional pixel constraints
        processor_kwargs = {}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        if cache_dir:
            processor_kwargs["cache_dir"] = cache_dir
            
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            **processor_kwargs
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
        else:
            # Eager attention for language model (needed for attention weight extraction)
            model_kwargs["attn_implementation"] = "eager"
            logger.info("Using eager attention for language model layers")
        
        model_kwargs.update(self.model_kwargs_extra)
        
        # Load model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # When using eager attention (for VAR/HGAI), switch the vision encoder
        # to SDPA to avoid OOM on high-resolution images. The vision encoder's
        # eager attention materializes a full attention matrix that can exceed
        # 59 GiB for high-res scorecard images. SDPA is memory-efficient and
        # built into PyTorch — no flash_attn package needed. Only the language
        # model layers need eager attention to return attention weights.
        if not self.use_flash_attention and hasattr(self.model, 'visual'):
            # Patch the vision encoder's config to use sdpa
            if hasattr(self.model.visual, 'config'):
                self.model.visual.config._attn_implementation = "sdpa"
            # Also patch each block's attention module directly
            for block in self.model.visual.blocks:
                if hasattr(block, 'attn'):
                    block.attn._attn_implementation = "sdpa"
            logger.info("Vision encoder switched to SDPA attention (memory-efficient)")
        
        logger.info(f"Model loaded successfully on device: {self.device}")
        logger.info(f"Quantization mode: {self.quantization}")
    
    def _extract_images_from_messages(self, messages: List[Dict]) -> Optional[List]:
        """
        Extract images from messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of images or None
        """
        images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img_data = item.get("image")
                        if img_data is not None:
                            if isinstance(img_data, Image.Image):
                                images.append(img_data)
                            elif isinstance(img_data, str):
                                # Could be URL, file path, or base64
                                try:
                                    images.append(load_image(img_data))
                                except Exception as e:
                                    logger.warning(f"Failed to load image: {e}")
        
        return images if images else None

    def raw_generate(self, 
                     text: str,
                     images: Optional[Union[Image.Image, List[Image.Image], str, List[str]]] = None,
                     max_new_tokens: int = 512,
                     **kwargs) -> str:
        """
        Generate text response without applying a chat template.

        Args:
            text: Raw text prompt to use directly (no chat template applied)
            images: Single image or list of images (PIL Images, URLs, file paths)
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Prepare images
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            # Load images if they're strings (URLs/paths)
            processed_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    processed_images.append(img)
                elif isinstance(img, str):
                    processed_images.append(load_image(img))
                else:
                    processed_images.append(img)
            images = processed_images
            
            inputs = self.processor(
                text=[text], 
                images=images, 
                padding=True, 
                return_tensors="pt"
            )
        else:
            inputs = self.processor(text=[text], padding=True, return_tensors="pt")
            
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                **kwargs
            )
        
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode response
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        return output_text[0]

    def generate(self, 
                 messages: List[Dict],
                 max_new_tokens: int = 512,
                 **kwargs) -> str:
        """
        Generate text response using the proper message format.
        
        Messages should follow the Qwen format:
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "path/url/base64"},
                    {"type": "text", "text": "Describe this image."}
                ]
            }
        ]
        
        Args:
            messages: List of message dictionaries with role and content
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Extract images from messages
        image_inputs = self._extract_images_from_messages(messages)
        
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                **kwargs
            )
        
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode response
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        return output_text[0]
    
    def chat(self, 
             user_message: str,
             images: Optional[Union[Image.Image, List[Image.Image], str, List[str]]] = None,
             system_prompt: Optional[str] = None,
             **generation_kwargs) -> str:
        """
        Chat interface for instruction-tuned model.
        
        Args:
            user_message: User's message/question
            images: Optional images (PIL Images, URLs, file paths, base64)
            system_prompt: Optional system prompt to guide behavior
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Model's response
        """
        # Build message content
        content = []
        
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            # Add image entries
            for img in images:
                if isinstance(img, Image.Image):
                    content.append({"type": "image", "image": img})
                else:
                    # URLs, file paths, base64 strings
                    content.append({"type": "image", "image": img})
        
        content.append({"type": "text", "text": user_message})
        
        # Create messages in the proper format
        messages = []
        if system_prompt:
            messages.append({
                "role": "system", 
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        messages.append({"role": "user", "content": content})
        
        return self.generate(messages, **generation_kwargs)
    
    def describe_image(self, 
                       image: Union[Image.Image, str],
                       question: str = "Describe this image in detail.",
                       **generation_kwargs) -> str:
        """
        Describe an image or answer questions about it.
        
        Args:
            image: PIL Image, URL, file path, or base64 string
            question: Question about the image
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Model's description/answer
        """
        return self.chat(question, images=image, **generation_kwargs)
    
    def load_image_from_url(self, url: str) -> Image.Image:
        """Load image from URL using transformers utility."""
        return load_image(url)
    
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
        }
        
        # Add memory usage if available
        if torch.cuda.is_available():
            info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        
        return info
    
    def __repr__(self):
        return f"Qwen25Wrapper2(model={self.model_name}, quantization_4bit={self.quantization_4bit}, flash_attn={self.use_flash_attention})"
