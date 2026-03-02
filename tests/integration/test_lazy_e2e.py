import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.core.lazy_handler import LazyHandlerProxy
from app.core.model_registry import ModelRegistry
from app.config import MultiModelServerConfig, ModelEntryConfig


@pytest.mark.anyio
async def test_lazy_load_unload_cycle() -> None:
    """End-to-end cycle test for LazyHandlerProxy within a ModelRegistry."""
    registry = ModelRegistry()
    model_id = "lazy-cycle-test"

    # Configure with very short idle timeout for testing
    proxy = LazyHandlerProxy(
        model_cfg_dict={"model_path": "test", "model_id": model_id},
        model_type="lm",
        model_path="test",
        model_id=model_id,
        idle_timeout_seconds=1,
    )
    await proxy.initialize()

    await registry.register_model(model_id, proxy, "lm")

    assert proxy.status == "unloaded"

    mock_handler = AsyncMock()
    mock_handler.generate_text_response = AsyncMock(
        return_value={"choices": [{"text": "hello"}]}
    )
    mock_handler.cleanup = AsyncMock()

    with patch(
        "app.core.handler_process.HandlerProcessProxy", return_value=mock_handler
    ):
        # 1. First request triggers load
        handler = registry.get_handler(model_id)
        response = await handler.generate_text_response({"prompt": "hi"})

        assert response == {"choices": [{"text": "hello"}]}
        assert proxy.status == "ready"
        assert proxy._handler is not None

        # 2. Wait for idle timeout
        # We need to wait slightly more than 1 second
        await asyncio.sleep(1.5)

        assert proxy.status == "unloaded"
        assert proxy._handler is None
        mock_handler.cleanup.assert_called_once()

        # 3. Subsequent request triggers re-load
        mock_handler.cleanup.reset_mock()
        response2 = await handler.generate_text_response({"prompt": "hi again"})

        assert response2 == {"choices": [{"text": "hello"}]}
        assert proxy.status == "ready"
        assert proxy._handler is not None


def test_lazy_config_verification_fallback() -> None:
    """Verify that the configuration correctly supports lazy loading fields."""
    config = MultiModelServerConfig(
        host="0.0.0.0",
        port=8000,
        default_lazy_load=True,
        default_idle_timeout_seconds=300,
        models=[
            ModelEntryConfig(
                model_path="test/model",
                model_id="test-model",
                model_type="lm",
                lazy_load=False,
                idle_timeout_seconds=600,
            )
        ],
    )

    assert config.default_lazy_load is True
    assert config.default_idle_timeout_seconds == 300
    assert config.models[0].lazy_load is False
    assert config.models[0].idle_timeout_seconds == 600
