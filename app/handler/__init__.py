"""
MLX model handlers for text, multimodal, image generation, and embeddings models.
"""

from .mlx_embeddings import MLXEmbeddingsHandler
from .mlx_lm import MLXLMHandler
from .mlx_vlm import MLXVLMHandler
from .mlx_speech import MLXSpeechHandler
from .mlx_tts import MLXTTSHandler

# Optional mflux import - only available if flux extra is installed
try:
    from .mflux import MLXFluxHandler

    MFLUX_AVAILABLE = True
except ImportError:
    MLXFluxHandler = None
    MFLUX_AVAILABLE = False

__all__ = [
    "MLXLMHandler",
    "MLXVLMHandler",
    "MLXFluxHandler",
    "MLXEmbeddingsHandler",
    "MLXSpeechHandler",
    "MLXTTSHandler",
    "MFLUX_AVAILABLE",
]
