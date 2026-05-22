import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image
from PIL import Image
import logging
from typing import List, Union, Optional, Dict, Any

logger = logging.getLogger(__name__)


class SmolVLMWrapper:
    """
    Wrapper for SmolVLM model with 4-bit quantization support.
    Loads the latest instruction-tuned version of SmolVLM.
    """
    
    def __init__(self, 
                 model_name: str = "HuggingFaceTB/SmolVLM-Instruct", 
                 quantization_4bit: bool = True,
                 device_map: str = "auto",
                 quantization: Optional[str] = None):
        """
        Initialize the SmolVLM wrapper.
        
        Args:
            model_name: HuggingFace model identifier for SmolVLM Instruct
            quantization_4bit: Whether to use 4-bit quantization.
                Deprecated in favor of `quantization`. Ignored when `quantization` is set.
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
        self.device_map = device_map
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.processor = None
        
        self._load_model()
        
    def _load_model(self):
        """Load the model, processor, and tokenizer with quantization if specified."""
        logger.info(f"Loading SmolVLM model: {self.model_name}")
        
        # Configure quantization if enabled
        quantization_config = None
        if self.quantization == "nf4":
            logger.info("Configuring 4-bit NF4 quantization with BitsAndBytes")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        elif self.quantization == "8bit":
            logger.info("Configuring 8-bit quantization with BitsAndBytes")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            logger.info("No quantization (full precision)")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Load model with quantization
        model_kwargs = {
            # "torch_dtype": torch.bfloat16,
            "dtype": torch.bfloat16,
            "_attn_implementation": "flash_attention_2" if self.device == "cuda" else "eager",
        }
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["device_map"] = self.device_map
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            **model_kwargs
        ).to(self.device)
        
        logger.info(f"Model loaded successfully on device: {self.model.device}")
        logger.info(f"Quantization mode: {self.quantization}")
    
    def raw_generate(self, 
                     text: str,
                     images: Optional[Union[Image.Image, List[Image.Image]]] = None,
                     max_new_tokens: int = 512,
                     **kwargs) -> str:
        """
        Generate text response without applying a chat template.

        Args:
            text: Raw text prompt to use directly (no chat template applied)
            images: Single image or list of images (PIL Images)
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Prepare inputs without applying chat template
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            inputs = self.processor(text=text, images=images, return_tensors="pt")
        else:
            inputs = self.processor(text=text, return_tensors="pt")
            
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
        
        # Decode response
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        return generated_texts[0]

    def generate(self, 
                 messages: List[Dict],
                 images: Optional[Union[Image.Image, List[Image.Image]]] = None,
                 max_new_tokens: int = 512,
                 **kwargs) -> str:
        """
        Generate text response using the proper message format.
        
        Args:
            messages: List of message dictionaries with role and content
            images: Single image or list of images (PIL Images) 
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Apply chat template
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Prepare inputs
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        else:
            inputs = self.processor(text=prompt, return_tensors="pt")
            
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
        
        # Decode response
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        return generated_texts[0]
    
    def chat(self, 
             user_message: str,
             images: Optional[Union[Image.Image, List[Image.Image]]] = None,
             system_prompt: Optional[str] = None,
             **generation_kwargs) -> str:
        """
        Chat interface for instruction-tuned model.
        
        Args:
            user_message: User's message/question
            images: Optional images to include in the conversation
            system_prompt: Optional system prompt to guide behavior
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Model's response
        """
        # Build message content
        content = []
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            # Add image placeholders
            for _ in images:
                content.append({"type": "image"})
        
        content.append({"type": "text", "text": user_message})
        
        # Create messages in the proper format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        
        messages.append({"role": "user", "content": content})
        
        return self.generate(messages, images=images, **generation_kwargs)
    
    def describe_image(self, 
                      image: Image.Image,
                      question: str = "Describe this image in detail.",
                      **generation_kwargs) -> str:
        """
        Describe an image or answer questions about it.
        
        Args:
            image: PIL Image to analyze
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
            "device": str(self.model.device),
            "model_dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "Unknown",
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        # Add memory usage if available
        if torch.cuda.is_available() and "cuda" in str(self.model.device):
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            info["gpu_memory_cached"] = torch.cuda.memory_reserved()
        
        return info
    
    def __repr__(self):
        return f"SmolVLMWrapper(model={self.model_name}, quantization_4bit={self.quantization_4bit})"
