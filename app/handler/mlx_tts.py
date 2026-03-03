import time
import uuid
import io
import numpy as np
from http import HTTPStatus
from typing import Any, Dict, List, Optional
import soundfile as sf

from loguru import logger
from fastapi import HTTPException
from fastapi.responses import Response

from ..core import InferenceWorker
from ..models.mlx_tts import MLXTTS
from ..schemas.openai import SpeechRequest
from ..utils.errors import create_error_response


class MLXTTSHandler:
    """
    Handler class for making requests to the underlying MLX TTS model service.
    Provides request queuing and audio generation handling.
    """

    handler_type: str = "tts"

    def __init__(self, model_path: str, max_concurrency: int = 1):
        """
        Initialize the handler with the specified model path.
        """
        self.model_path = model_path
        self.model = MLXTTS(model_path)
        self.model_created = int(time.time())
        self.inference_worker = InferenceWorker()
        logger.info(f"Initialized MLXTTSHandler with model path: {model_path}")

    async def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with their metadata."""
        try:
            return [{
                "id": self.model_path,
                "object": "model",
                "created": self.model_created,
                "owned_by": "local"
            }]
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return []

    async def initialize(self, queue_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the handler and start the inference worker."""
        if not queue_config:
            queue_config = {
                "timeout": 600,
                "queue_size": 50,
            }
        self.inference_worker = InferenceWorker(
            queue_size=queue_config.get("queue_size", 50),
            timeout=queue_config.get("timeout", 600),
        )
        self.inference_worker.start()
        logger.info("Initialized MLXTTSHandler and started inference worker")

    async def generate_speech_response(self, request: SpeechRequest) -> dict[str, Any]:
        """
        Generate a TTS audio response for the given request.
        """
        request_id = f"tts-{uuid.uuid4()}"

        try:
            # Submit to the inference thread
            result = await self.inference_worker.submit(
                self.model.generate_speech,
                text=request.input,
                voice=request.voice,
                speed=request.speed,
            )

            audio_data: np.ndarray = result["audio"]
            sample_rate: int = result["sample_rate"]

            format_mapping = {
                "mp3": "mp3", # soundfile handles some, but let's default to basic formats or whatever soundfile supports
                "wav": "wav",
                "flac": "flac",
                "opus": "ogg",
                "aac": "flac", # Not directly supported by soundfile, fallback
                "pcm": "raw"
            }
            file_format = format_mapping.get(request.response_format, "wav")

            buffer = io.BytesIO()
            # Soundfile writes to buffer
            sf.write(buffer, audio_data, sample_rate, format=file_format.upper())
            
            buffer.seek(0)
            audio_bytes = buffer.read()

            media_type_map = {
                "wav": "audio/wav",
                "mp3": "audio/mpeg",
                "flac": "audio/flac",
                "opus": "audio/ogg",
                "pcm": "audio/l16",
            }
            media_type = media_type_map.get(request.response_format, f"audio/{file_format}")

            return {
                "content": audio_bytes,
                "media_type": media_type
            }

        except Exception as e:
            logger.exception(f"Error in MLXTTSHandler: {str(e)}")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=str(e)
            ) from e
