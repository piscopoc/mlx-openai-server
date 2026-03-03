import numpy as np
from typing import Any, Dict
import mlx.core as mx
from mlx_audio.tts.utils import load_model

class MLXTTS:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = load_model(self.model_path)

    def generate_speech(self, text: str, voice: str, speed: float) -> Dict[str, Any]:
        """
        Generate audio from text using mlx-audio tts model.
        Returns a dictionary with 'audio' (numpy array) and 'sample_rate' (int).
        """
        results = list(self.model.generate(text=text, voice=voice, speed=speed))
        
        # Concatenate audio chunks
        audio_chunks = [result.audio for result in results]
        
        if not audio_chunks:
            raise ValueError("Model produced no audio output")
            
        if len(audio_chunks) > 1:
            full_audio = mx.concatenate(audio_chunks, axis=-1)
        else:
            full_audio = audio_chunks[0]

        # Extract sample rate from the first result
        sample_rate = results[0].sample_rate

        # Ensure valid length if batch dim needs removal
        if len(full_audio.shape) > 1 and full_audio.shape[0] == 1:
            full_audio = full_audio[0]

        audio_np = np.array(full_audio)

        return {
            "audio": audio_np,
            "sample_rate": sample_rate
        }
