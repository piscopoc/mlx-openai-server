"""Tests for the LazyHandlerProxy."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.lazy_handler import LazyHandlerProxy


@pytest.mark.anyio
async def test_lazy_handler_initialization() -> None:
    """Test that LazyHandlerProxy initializes correctly without spawning a handler."""
    model_cfg_dict = {"model_path": "test", "model_id": "test", "model_type": "lm"}
    proxy = LazyHandlerProxy(
        model_cfg_dict=model_cfg_dict,
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=300,
    )
    assert proxy._handler is None
    assert proxy._last_activity == 0
    assert proxy._idle_timeout_seconds == 300
    assert proxy.status == "unloaded"


@pytest.mark.anyio
async def test_spawn_on_first_call() -> None:
    """Test that the handler is spawned on the first request."""
    model_cfg_dict = {"model_path": "test", "model_id": "test", "model_type": "lm"}
    proxy = LazyHandlerProxy(
        model_cfg_dict=model_cfg_dict,
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=300,
    )
    await proxy.initialize()

    mock_handler_instance = AsyncMock()
    mock_handler_instance.start = AsyncMock()
    mock_handler_instance.generate_text_response = AsyncMock(return_value={"response": "test"})

    with patch(
        "app.core.handler_process.HandlerProcessProxy", return_value=mock_handler_instance
    ) as MockHandler:
        response = await proxy.generate_text_response({"messages": []})

        assert response == {"response": "test"}
        MockHandler.assert_called_once_with(
            model_cfg_dict=model_cfg_dict, model_type="lm", model_path="test", model_id="test"
        )
        mock_handler_instance.start.assert_called_once()
        assert proxy._handler == mock_handler_instance
        assert proxy.status == "ready"


@pytest.mark.anyio
async def test_reuses_handler_on_subsequent_calls() -> None:
    """Test that the handler is reused for subsequent requests."""
    model_cfg_dict = {"model_path": "test", "model_id": "test", "model_type": "lm"}
    proxy = LazyHandlerProxy(
        model_cfg_dict=model_cfg_dict,
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=300,
    )
    await proxy.initialize()

    mock_handler = AsyncMock()
    mock_handler.generate_text_response = AsyncMock(return_value={"response": "test"})
    proxy._handler = mock_handler

    with patch.object(proxy, "_reset_idle_timer") as mock_reset:
        await proxy.generate_text_response({"messages": []})
        await proxy.generate_text_response({"messages": []})

        # Verify handler methods were called
        assert mock_handler.generate_text_response.call_count == 2
        # Verify timer was reset for each call
        assert mock_reset.call_count == 2


@pytest.mark.anyio
async def test_idle_timer_resets_on_activity() -> None:
    """Test that the idle timer correctly resets last_activity and cancels existing unload task."""
    proxy = LazyHandlerProxy(
        model_cfg_dict={},
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=300,
    )
    proxy._handler = Mock()
    mock_task = Mock()
    mock_task.done.return_value = False
    proxy._unload_task = mock_task

    initial_activity = 1000.0
    proxy._last_activity = initial_activity

    with patch("time.time", return_value=2000.0):
        proxy._reset_idle_timer()

        assert proxy._last_activity == 2000.0
        mock_task.cancel.assert_called_once()


@pytest.mark.anyio
async def test_concurrent_requests_during_spawn() -> None:
    """Test that concurrent requests wait for the same handler to spawn."""
    model_cfg_dict = {"model_path": "test", "model_id": "test", "model_type": "lm"}
    proxy = LazyHandlerProxy(
        model_cfg_dict=model_cfg_dict,
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=300,
    )
    await proxy.initialize()

    mock_handler = AsyncMock()

    # Simulate some delay in starting
    async def slow_start(*args, **kwargs):
        await asyncio.sleep(0.1)

    mock_handler.start = AsyncMock(side_effect=slow_start)
    mock_handler.generate_text_response = AsyncMock(return_value={"response": "test"})

    with patch(
        "app.core.handler_process.HandlerProcessProxy", return_value=mock_handler
    ) as MockHandler:
        # Launch multiple concurrent requests
        results = await asyncio.gather(
            proxy.generate_text_response({"messages": []}),
            proxy.generate_text_response({"messages": []}),
            proxy.generate_text_response({"messages": []}),
        )

        assert len(results) == 3
        assert all(r == {"response": "test"} for r in results)

        # HandlerProcessProxy should only be instantiated once
        assert MockHandler.call_count == 1
        # start() should only be called once
        assert mock_handler.start.call_count == 1


@pytest.mark.anyio
async def test_concurrent_during_unload() -> None:
    """Test that requests during or after unload re-spawn the handler correctly."""
    model_cfg_dict = {"model_path": "test", "model_id": "test", "model_type": "lm"}
    proxy = LazyHandlerProxy(
        model_cfg_dict=model_cfg_dict,
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=0.1,  # Short timeout
    )
    await proxy.initialize()

    mock_handler1 = AsyncMock()
    mock_handler1.generate_text_response = AsyncMock(return_value={"resp": 1})
    mock_handler1.cleanup = AsyncMock()

    mock_handler2 = AsyncMock()
    mock_handler2.generate_text_response = AsyncMock(return_value={"resp": 2})

    with patch("app.core.handler_process.HandlerProcessProxy") as MockHandler:
        MockHandler.side_effect = [mock_handler1, mock_handler2]

        # First call spawns mock_handler1
        resp1 = await proxy.generate_text_response({})
        assert resp1 == {"resp": 1}
        assert proxy._handler == mock_handler1

        # Wait for idle unload
        await asyncio.sleep(0.2)
        assert proxy._handler is None
        mock_handler1.cleanup.assert_called_once()

        # Second call spawns mock_handler2
        resp2 = await proxy.generate_text_response({})
        assert resp2 == {"resp": 2}
        assert proxy._handler == mock_handler2
