import pytest
from app.schemas.openai import SpeechRequest

def test_speech_request_schema():
    # Test valid request
    data = {
        "model": "qwen3-tts",
        "input": "Hello world",
        "voice": "alloy",
        "response_format": "wav",
        "speed": 1.5
    }
    request = SpeechRequest(**data)
    assert request.model == "qwen3-tts"
    assert request.input == "Hello world"
    assert request.voice == "alloy"
    assert request.response_format == "wav"
    assert request.speed == 1.5

def test_speech_request_defaults():
    data = {
        "model": "qwen3-tts",
        "input": "Hello default"
    }
    request = SpeechRequest(**data)
    assert request.voice == "alloy"
    assert request.response_format == "wav"
    assert request.speed == 1.0

def test_speech_request_invalid_speed():
    data = {
        "model": "qwen3-tts",
        "input": "Hello world",
        "speed": 5.0 # Max is 4.0
    }
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SpeechRequest(**data)
