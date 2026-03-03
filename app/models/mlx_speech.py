import librosa
import numpy as np
from functools import lru_cache
from mlx_audio.stt.utils import load

SAMPLING_RATE = 16000
CHUNK_SIZE = 30


@lru_cache(maxsize=32)
def load_audio(fname):
    """Load and cache audio file. Cache size limited to 32 recent files."""
    a, _ = librosa.load(fname, sr=SAMPLING_RATE, dtype=np.float32)
    return a

@lru_cache(maxsize=32)
def calculate_audio_duration(audio_path: str) -> int:
    """Calculate the duration of the audio file in seconds."""
    audio = load_audio(audio_path)
    return len(audio) / SAMPLING_RATE

class MLXSpeech:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = load(self.model_path)

    def _transcribe_generator(self, audio_path: str, **kwargs):
        """Stream transcription by yielding dicts with text chunks."""
        # Using mlx-audio generate with stream=True
        # According to mlx-audio documentation, stream=True yields chunks or text.
        for chunk in self.model.generate(audio_path, stream=True, **kwargs):
            # Check if chunk is a string or an object with .text
            if isinstance(chunk, str):
                text = chunk
            elif hasattr(chunk, "text"):
                text = chunk.text
            else:
                text = str(chunk)

            yield {"text": text}

    def __call__(self, audio_path: str, stream: bool = False, **kwargs):
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            stream: If True, yields chunks. If False, transcribes entire file at once.
            **kwargs: Additional arguments passed to generate()
        """
        if stream:
            return self._transcribe_generator(audio_path, **kwargs)
        else:
            result = self.model.generate(audio_path, **kwargs)
            if isinstance(result, str):
                text = result
            elif hasattr(result, "text"):
                text = result.text
            elif isinstance(result, dict) and "text" in result:
                text = result["text"]
            else:
                text = str(result)
            return {"text": text}


if __name__ == "__main__":
    model = MLXSpeech("mlx-community/whisper-tiny")
    # Non-streaming (fastest for most use cases)
    result = model("examples/audios/podcast.wav", stream=True)
    for chunk in result:
        print(f"text: {chunk['text']}")
