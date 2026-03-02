# Lazy Model Loading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable lazy loading to multi-handler mode, spawning models on first request and auto-unloading after configurable idle periods to reduce memory footprint.

**Architecture:** Create `LazyHandlerProxy` wrapper around existing `HandlerProcessProxy` that manages on-demand spawning, activity tracking, and idle timer-based auto-unload. Extend YAML configuration schema with optional `lazy_load`, `idle_timeout_seconds`, and `preload` fields per model entry, maintaining full backward compatibility.

**Tech Stack:** Python 3.11+, asyncio, multiprocessing, Pydantic, pytest, FastAPI

**Upstream Repository:** https://github.com/piscopoc/mlx-openai-server

**Important:** Do NOT commit any changes to source control. This is a development fork.

---

## Task 1: Extend configuration schema

**Files:**
- Modify: `app/config.py`

**Step 1: Add lazy loading fields to ModelEntryConfig**

```python
# In ModelEntryConfig dataclass, add:
lazy_load: bool = False
idle_timeout_seconds: int = 0
preload: bool = False
```

**Step 2: Add server-level defaults to MultiModelServerConfig**

```python
# In MultiModelServerConfig dataclass, add:
default_lazy_load: bool = False
default_idle_timeout_seconds: int = 1800
```

**Step 3: Add __post_init__ to apply server-level defaults**

```python
def __post_init__(self):
    for model in self.models:
        if model.lazy_load is False:  # Uses default, apply server default
            model.lazy_load = self.default_lazy_load
        if model.idle_timeout_seconds == 0:  # Uses default, apply server default
            model.idle_timeout_seconds = self.default_idle_timeout_seconds
```

**Step 4: Run existing tests to ensure backward compatibility**

Run: `pytest tests/ -k config -v`
Expected: All existing tests pass (new fields have safe defaults)

**Note:** Do NOT commit changes to source control.

---

## Task 2: Create LazyHandlerProxy class

**Files:**
- Create: `app/core/lazy_handler.py`

**Step 1: Write failing test for LazyHandlerProxy structure**

```python
# tests/core/test_lazy_handler.py
import pytest
from app.core.lazy_handler import LazyHandlerProxy

def test_lazy_handler_initialization():
    proxy = LazyHandlerProxy(
        model_cfg_dict={"model_path": "test", "model_id": "test", "model_type": "lm"},
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=300,
    )
    assert proxy._handler is None
    assert proxy._last_activity == 0
    assert proxy._idle_timeout_seconds == 300
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_lazy_handler.py::test_lazy_handler_initialization -v`
Expected: FAIL with "LazyHandlerProxy not found"

**Step 3: Write LazyHandlerProxy class skeleton**

```python
# app/core/lazy_handler.py
import asyncio
import time
import threading
from typing import Any, AsyncGenerator
from loguru import logger

from app.core.handler_process import HandlerProcessProxy


class LazyHandlerProxy:
    """Wrapper for HandlerProcessProxy with lazy loading lifecycle.

    Spawns handler on first request, tracks activity for idle detection,
    and auto-unloads after configured timeout period.
    """

    def __init__(
        self,
        model_cfg_dict: dict[str, Any],
        model_type: str,
        model_path: str,
        model_id: str,
        idle_timeout_seconds: int = 0,
    ) -> None:
        self.model_path = model_path
        self.model_id = model_id
        self.handler_type = model_type
        self._model_cfg_dict = model_cfg_dict
        self._idle_timeout_seconds = idle_timeout_seconds

        # Handler lazily created
        self._handler: HandlerProcessProxy | None = None
        self._last_activity: float = 0
        self._unload_task: asyncio.Task | None = None
        self._loading_lock = asyncio.Lock()
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._model_created: int = 0

    def _reset_idle_timer(self) -> None:
        """Reset idle timer on activity."""
        self._last_activity = time.time()
        if self._unload_task and not self._unload_task.done():
            self._unload_task.cancel()

    def _start_idle_timer_if_needed(self) -> None:
        """Start idle timer if timeout is configured."""
        if self._idle_timeout_seconds <= 0:
            return
        self._unload_task = asyncio.create_task(self._idle_unload())

    async def _idle_unload(self) -> None:
        """Wait for idle timeout and unload handler."""
        delay = self._idle_timeout_seconds
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return

        time_since_last = time.time() - self._last_activity
        if time_since_last >= delay and self._handler is not None:
            await self._unload_handler()

    async def _unload_handler(self) -> None:
        """Unload handler process."""
        logger.info(f"Idle timeout for model '{self.model_id}', unloading...")
        if self._handler:
            try:
                await self._handler.cleanup()
            except Exception as e:
                logger.error(f"Error unloading handler: {e}")
            finally:
                self._handler = None
                self._model_created = 0
                logger.info(f"Model '{self.model_id}' unloaded")

    async def _ensure_handler(self, queue_config: dict[str, Any]) -> HandlerProcessProxy:
        """Ensure handler is loaded, spawning if necessary."""
        if self._handler is not None:
            return self._handler

        async with self._loading_lock:
            # Double-check after acquiring lock
            if self._handler is not None:
                return self._handler

            logger.info(f"Spawning handler on-demand for model '{self.model_id}'...")
            from app.core.handler_process import HandlerProcessProxy

            proxy = HandlerProcessProxy(
                model_cfg_dict=self._model_cfg_dict,
                model_type=self.handler_type,
                model_path=self.model_path,
                model_id=self.model_id,
            )
            await proxy.start(queue_config)
            self._handler = proxy
            self._model_created = int(time.time())
            self._last_activity = time.time()
            self._start_idle_timer_if_needed()
            logger.info(f"On-demand handler for '{self.model_id}' ready")

        return self._handler

    async def initialize(
        self, queue_config: dict[str, Any] | None = None
    ) -> None:
        """No-op — handler loads on first request."""
        self._loop = asyncio.get_running_loop()
        self._running = True

    async def cleanup(self) -> None:
        """Clean up handler if loaded."""
        self._running = False
        if self._handler:
            await self._handler.cleanup()
            self._handler = None
        if self._unload_task and not self._unload_task.done():
            self._unload_task.cancel()

    # Placeholder methods matching HandlerProcessProxy interface
    async def get_models(self) -> list[dict[str, Any]]:
        return [{"id": self.model_id, "object": "model"}]

    async def get_queue_stats(self) -> dict[str, Any]:
        return {"queue_size": 0, "active_requests": 0}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_lazy_handler.py::test_lazy_handler_initialization -v`
Expected: PASS

**Note:** Do NOT commit changes to source control.

---

## Task 3: Implement request forwarding in LazyHandlerProxy

**Files:**
- Modify: `app/core/lazy_handler.py`

**Step 1: Write failing test for request forwarding**

```python
def test_request_spawns_handler():
    proxy = LazyHandlerProxy(
        model_cfg_dict={"model_path": "test", "model_id": "test", "model_type": "lm"},
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=300,
    )
    proxy._loop = asyncio.new_event_loop()

    async def test_spawn():
        mock_handler = Mock()
        proxy._handler = mock_handler
        return

    # Mock _ensure_handler to avoid real spawn
    original_ensure = proxy._ensure_handler
    async def mock_ensure(qc):
        proxy._handler = Mock()
        return proxy._handler
    proxy._ensure_handler = mock_ensure

    async def run():
        await proxy._ensure_handler({})
        assert proxy._handler is not None

    asyncio.run(run())
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_lazy_handler.py::test_request_spawns_handler -v`
Expected: FAIL (implementation incomplete)

**Step 3: Implement generate_text_stream method**

```python
async def generate_text_stream(
    self, request: Any
) -> AsyncGenerator[Any, None]:
    """Forward streaming text generation, spawning handler if needed."""
    handler = await self._ensure_handler({})
    self._reset_idle_timer()
    self._start_idle_timer_if_needed()

    if hasattr(handler, 'generate_text_stream'):
        async for chunk in handler.generate_text_stream(request):
            self._reset_idle_timer()  # Update on each chunk
            yield chunk
    else:
        raise AttributeError("Handler does not support generate_text_stream")
```

**Step 4: Implement generate_text_response method**

```python
async def generate_text_response(self, request: Any) -> dict[str, Any]:
    """Forward non-streaming text generation, spawning handler if needed."""
    handler = await self._ensure_handler({})
    self._reset_idle_timer()
    self._start_idle_timer_if_needed()
    return await handler.generate_text_response(request)
```

**Step 5: Implement remaining handler methods**

```python
async def generate_multimodal_stream(
    self, request: Any
) -> AsyncGenerator[Any, None]:
    handler = await self._ensure_handler({})
    self._reset_idle_timer()
    self._start_idle_timer_if_needed()
    async for chunk in handler.generate_multimodal_stream(request):
        self._reset_idle_timer()
        yield chunk

async def generate_multimodal_response(self, request: Any) -> dict[str, Any]:
    handler = await self._ensure_handler({})
    self._reset_idle_timer()
    self._start_idle_timer_if_needed()
    return await handler.generate_multimodal_response(request)

async def generate_embeddings_response(self, request: Any) -> Any:
    handler = await self._ensure_handler({})
    self._reset_idle_timer()
    self._start_idle_timer_if_needed()
    return await handler.generate_embeddings_response(request)

async def generate_image(self, request: Any) -> Any:
    handler = await self._ensure_handler({})
    self._reset_idle_timer()
    self._start_idle_timer_if_needed()
    return await handler.generate_image(request)

async def edit_image(self, request: Any) -> Any:
    handler = await self._ensure_handler({})
    self._reset_idle_timer()
    self._start_idle_timer_if_needed()
    return await handler.edit_image(request)

async def prepare_transcription_request(self, request: Any) -> dict[str, Any]:
    handler = await self._ensure_handler({})
    self._reset_idle_timer()
    self._start_idle_timer_if_needed()
    return await handler.prepare_transcription_request(request)

async def generate_transcription_response(self, request: Any) -> Any:
    handler = await self._ensure_handler({})
    self._reset_idle_timer()
    self._start_idle_timer_if_needed()
    return await handler.generate_transcription_response(request)

async def generate_transcription_stream_from_data(
    self, request_data: dict[str, Any], *args: Any, **kwargs: Any
) -> AsyncGenerator[str, None]:
    handler = await self._ensure_handler({})
    self._reset_idle_timer()
    self._start_idle_timer_if_needed()
    async for chunk in handler.generate_transcription_stream_from_data(request_data, *args, **kwargs):
        self._reset_idle_timer()
        yield chunk
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/core/test_lazy_handler.py -v`
Expected: PASS for request forwarding tests

**Note:** Do NOT commit changes to source control.

---

## Task 4: Modify server.py to use lazy proxies

**Files:**
- Modify: `app/server.py`

**Step 1: Write failing test for lazy handler creation**

```python
def test_multi_lifespan_creates_lazy_proxies():
    config = MultiModelServerConfig(
        host="0.0.0.0",
        port=8000,
        models=[
            ModelEntryConfig(
                model_path="test",
                model_id="lazy-model",
                model_type="lm",
                lazy_load=True,
                idle_timeout_seconds=300,
            )
        ]
    )

    async def test():
        from app.server import create_multi_lifespan
        lifespan = create_multi_lifespan(config)
        # Verify proxy type created
        # This would require mocking to handler creation

    asyncio.run(test())
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_multi_lifespan_creates_lazy_proxies -v`
Expected: FAIL (not implemented yet)

**Step 3: Import LazyHandlerProxy in server.py**

```python
from app.core.lazy_handler import LazyHandlerProxy
```

**Step 4: Modify create_multi_lifespan to handle lazy/eager**

```python
async def lifespan(app: FastAPI):
    registry = ModelRegistry()

    try:
        for model_cfg in config.models:
            model_id = model_cfg.model_id
            logger.info(
                f"Processing model '{model_id}' "
                f"(type={model_cfg.model_type}, lazy={model_cfg.lazy_load}, preload={model_cfg.preload})"
            )

            queue_config = {
                "max_concurrency": model_cfg.max_concurrency,
                "timeout": model_cfg.queue_timeout,
                "queue_size": model_cfg.queue_size,
            }

            # Eager load (preload=True or lazy_load=False)
            if model_cfg.preload or not model_cfg.lazy_load:
                from dataclasses import asdict
                model_cfg_dict = asdict(model_cfg)

                logger.info(f"Spawning handler (eager) for model '{model_id}'")
                proxy = HandlerProcessProxy(
                    model_cfg_dict=model_cfg_dict,
                    model_type=model_cfg.model_type,
                    model_path=model_cfg.model_path,
                    model_id=model_id,
                )
                await proxy.start(queue_config)
            # Lazy load - use wrapper
            else:
                from dataclasses import asdict
                model_cfg_dict = asdict(model_cfg)

                logger.info(f"Creating lazy proxy for model '{model_id}' (will spawn on first request)")
                proxy = LazyHandlerProxy(
                    model_cfg_dict=model_cfg_dict,
                    model_type=model_cfg.model_type,
                    model_path=model_cfg.model_path,
                    model_id=model_id,
                    idle_timeout_seconds=model_cfg.idle_timeout_seconds,
                )
                # Initialize lazy proxy (no spawn yet)
                await proxy.initialize()

            await registry.register_model(
                model_id=model_id,
                handler=proxy,
                model_type=model_cfg.model_type,
                context_length=model_cfg.context_length,
            )
            logger.info(f"Model '{model_id}' registered successfully")

        app.state.registry = registry

        if config.models:
            first_id = config.models[0].model_id
            app.state.handler = registry.get_handler(first_id)

        logger.info(
            f"Multi-handler initialization complete. "
            f"{registry.get_model_count()} model(s) registered."
        )

    except Exception as e:
        logger.error(f"Failed to initialize multi-handler setup: {e}")
        await registry.cleanup_all()
        raise

    mx.clear_cache()
    gc.collect()

    yield

    logger.info("Shutting down multi-handler application")
    await registry.cleanup_all()

    mx.clear_cache()
    gc.collect()
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_server.py -k multi_lifespan -v`
Expected: PASS

**Note:** Do NOT commit changes to source control.

---

## Task 5: Update models endpoint to show load status

**Files:**
- Modify: `app/api/endpoints.py`

**Step 1: Write failing test for status field**

```python
def test_models_endpoint_includes_status():
    # Mock registry with mixed loaded/unloaded models
    response = client.get("/v1/models")
    data = response.json()["data"]
    assert all("status" in model for model in data)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_endpoints.py::test_models_endpoint_includes_status -v`
Expected: FAIL (status field not present)

**Step 3: Add status property to LazyHandlerProxy**

```python
@property
def status(self) -> str:
    """Return current load status."""
    if self._handler is None:
        return "unloaded"
    return "ready"
```

**Step 4: Add status to HandlerProcessProxy for consistency**

```python
# In app/core/handler_process.py, add to HandlerProcessProxy:
@property
def status(self) -> str:
    """Return status - always ready for eager handlers."""
    return "ready"
```

**Step 5: Update models endpoint to include status**

```python
@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    """List all available models with their status."""
    registry: ModelRegistry = request.app.state.registry

    models_data = []
    for model_id, handler in registry._handlers.items():
        status = getattr(handler, "status", "unknown")
        model_info = {
            "id": model_id,
            "object": "model",
            "status": status,
        }
        models_data.append(model_info)

    return {"object": "list", "data": models_data}
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/api/test_endpoints.py::test_models_endpoint_includes_status -v`
Expected: PASS

**Note:** Do NOT commit changes to source control.

---

## Task 6: Update example config with lazy loading

**Files:**
- Modify: `examples/config.yaml`

**Step 1: Add lazy loading examples to config**

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  log_level: INFO
  default_lazy_load: true
  default_idle_timeout_seconds: 1800  # 30 minutes

models:
  # Frequently-used model - preload at startup
  - model_path: mlx-community/Qwen3-Coder-Next-4bit
    model_type: lm
    model_id: qwen3-coder
    lazy_load: false
    preload: true

  # Rarely-used model - lazy load
  - model_path: mlx-community/GLM-4.7-Flash-8bit
    model_type: lm
    model_id: glm-4.7-flash
    lazy_load: true
    idle_timeout_seconds: 3600  # 1 hour

  # Vision model - lazy load with short timeout
  - model_path: mlx-community/Qwen3-VL-2B-Instruct-4bit
    model_type: multimodal
    model_id: qwen3-vl
    lazy_load: true
    idle_timeout_seconds: 900  # 15 minutes

  # Image model - always loaded if used
  - model_path: black-forest-labs/FLUX.2-klein-4B
    model_type: image-generation
    config_name: flux2-klein-4b
    quantize: 4
    model_id: flux2-klein-4b
    lazy_load: false
```

**Note:** Do NOT commit changes to source control.

---

## Task 7: Update README with lazy loading documentation

**Files:**
- Modify: `README.md`

**Step 1: Add lazy loading section after "Launching Multiple Models"**

```markdown
## Lazy Model Loading

Models can be configured to load on-demand instead of at startup, reducing memory footprint for servers with many models.

### Configuration Options

Add these fields to your `config.yaml`:

```yaml
server:
  default_lazy_load: true          # Global default: enable lazy loading
  default_idle_timeout_seconds: 1800  # Global default: auto-unload after 30 min

models:
  - model_path: path/to/model
    model_id: my-model
    lazy_load: true              # Enable lazy loading (default: false)
    idle_timeout_seconds: 1800     # Auto-unload after N seconds (0 = never)
    preload: false                # Load at startup despite lazy_load (default: false)
```

**Field descriptions:**

| Field | Default | Description |
|-------|----------|-------------|
| `lazy_load` | `false` | Load model on first request instead of startup |
| `idle_timeout_seconds` | `0` | Unload after N seconds of inactivity. `0` = never unload |
| `preload` | `false` | Load at startup even with `lazy_load: true` (for critical models) |
| `default_lazy_load` | `false` | Server-level default for `lazy_load` |
| `default_idle_timeout_seconds` | `1800` | Server-level default for `idle_timeout_seconds` |

### Behavior

- **Lazy models**: Spawn on first request, auto-unload after idle period
- **Preload models**: Load at startup, no auto-unload (existing behavior)
- **Mixed config**: Supports both approaches in same config file
- **Status check**: `GET /v1/models` returns `"status": "ready"` or `"status": "unloaded"`

### Example Config

See `examples/config.yaml` for a complete example with lazy loading configured.
```

**Note:** Do NOT commit changes to source control.

---

## Task 8: Add integration tests for lazy loading

**Files:**
- Create: `tests/integration/test_lazy_loading.py`

**Step 1: Write failing test for cold start behavior**

```python
import pytest
from app.config import load_config_from_yaml, ModelEntryConfig, MultiModelServerConfig


@pytest.fixture
def lazy_config():
    return MultiModelServerConfig(
        host="0.0.0.0",
        port=8000,
        default_lazy_load=True,
        default_idle_timeout_seconds=60,
        models=[
            ModelEntryConfig(
                model_path="mlx-community/test-model",
                model_id="lazy-model",
                model_type="lm",
                lazy_load=True,
                idle_timeout_seconds=60,
            ),
            ModelEntryConfig(
                model_path="mlx-community/other-model",
                model_id="preload-model",
                model_type="lm",
                lazy_load=False,
                preload=True,
            ),
        ]
    )


def test_cold_start_only_preloads_marked_models(lazy_config):
    """Verify only preload models load at startup."""
    # In actual integration, would start server and check /v1/models
    # For now, verify config parsing
    assert lazy_config.models[0].lazy_load is True
    assert lazy_config.models[0].preload is False
    assert lazy_config.models[1].lazy_load is False
    assert lazy_config.models[1].preload is True
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/integration/test_lazy_loading.py::test_cold_start_only_preloads_marked_models -v`
Expected: PASS

**Step 3: Write test for idle timeout behavior**

```python
import time


def test_idle_timeout_configuration():
    """Verify idle timeout is configurable."""
    config = MultiModelServerConfig(
        host="0.0.0.0",
        port=8000,
        default_idle_timeout_seconds=1800,
        models=[
            ModelEntryConfig(
                model_path="test",
                model_id="short-timeout",
                model_type="lm",
                lazy_load=True,
                idle_timeout_seconds=300,  # 5 minutes
            ),
            ModelEntryConfig(
                model_path="test2",
                model_id="long-timeout",
                model_type="lm",
                lazy_load=True,
                idle_timeout_seconds=7200,  # 2 hours
            ),
        ]
    )
    assert config.models[0].idle_timeout_seconds == 300
    assert config.models[1].idle_timeout_seconds == 7200


def test_zero_timeout_disables_unload():
    """Verify zero timeout means never unload."""
    config = ModelEntryConfig(
        model_path="test",
        model_id="no-unload",
        model_type="lm",
        lazy_load=True,
        idle_timeout_seconds=0,
    )
    assert config.idle_timeout_seconds == 0
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/integration/test_lazy_loading.py -v`
Expected: PASS

**Note:** Do NOT commit changes to source control.

---

## Task 9: Add unit tests for LazyHandlerProxy

**Files:**
- Modify: `tests/core/test_lazy_handler.py`

**Step 1: Write test for handler spawn on first call**

```python
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from app.core.lazy_handler import LazyHandlerProxy


@pytest.mark.asyncio
async def test_spawn_on_first_call():
    """Verify handler spawns on first request."""
    proxy = LazyHandlerProxy(
        model_cfg_dict={"model_path": "test", "model_id": "test", "model_type": "lm"},
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=300,
    )
    await proxy.initialize()

    # Mock HandlerProcessProxy.start to avoid real spawn
    with patch('app.core.lazy_handler.HandlerProcessProxy') as MockHandler:
        mock_instance = Mock()
        mock_instance.start = AsyncMock()
        mock_instance.generate_text_response = AsyncMock(return_value={"response": "test"})
        MockHandler.return_value = mock_instance

        await proxy.generate_text_response({"messages": []})

        # Verify handler was created and started
        MockHandler.assert_called_once()
        mock_instance.start.assert_called_once()
        assert proxy._handler is not None


@pytest.mark.asyncio
async def test_reuses_handler_on_subsequent_calls():
    """Verify same handler is reused for subsequent requests."""
    proxy = LazyHandlerProxy(
        model_cfg_dict={"model_path": "test", "model_id": "test", "model_type": "lm"},
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=300,
    )
    await proxy.initialize()

    mock_handler = Mock()
    mock_handler.generate_text_response = AsyncMock(return_value={"response": "test"})

    proxy._handler = mock_handler
    proxy._reset_idle_timer = Mock()
    proxy._start_idle_timer_if_needed = Mock()

    await proxy.generate_text_response({"messages": []})
    await proxy.generate_text_response({"messages": []})

    # Verify handler reused, timer reset called twice
    assert mock_handler.generate_text_response.call_count == 2
    assert proxy._reset_idle_timer.call_count == 2
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/core/test_lazy_handler.py -v`
Expected: PASS

**Step 3: Write test for idle timer behavior**

```python
import time


@pytest.mark.asyncio
async def test_idle_timer_resets_on_activity():
    """Verify idle timer resets when handler is used."""
    proxy = LazyHandlerProxy(
        model_cfg_dict={"model_path": "test", "model_id": "test", "model_type": "lm"},
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=300,
    )
    await proxy.initialize()

    # Simulate initial activity
    initial_time = time.time()
    proxy._last_activity = initial_time
    proxy._unload_task = Mock()
    proxy._unload_task.done = Mock(return_value=False)

    # Reset timer
    proxy._reset_idle_timer()
    proxy._start_idle_timer_if_needed = Mock()

    # Verify last activity updated and old task cancelled
    assert proxy._last_activity > initial_time
    proxy._unload_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_zero_timeout_disables_idle():
    """Verify zero timeout doesn't start idle timer."""
    proxy = LazyHandlerProxy(
        model_cfg_dict={"model_path": "test", "model_id": "test", "model_type": "lm"},
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=0,  # Disabled
    )
    await proxy.initialize()

    # Reset should not start timer
    with patch('asyncio.create_task') as create_task:
        proxy._reset_idle_timer()
        proxy._start_idle_timer_if_needed()
        create_task.assert_not_called()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/core/test_lazy_handler.py -v`
Expected: PASS

**Note:** Do NOT commit changes to source control.

---

## Task 10: Test concurrent requests during spawn

**Files:**
- Modify: `tests/core/test_lazy_handler.py`

**Step 1: Write test for concurrent spawn protection**

```python
import asyncio


@pytest.mark.asyncio
async def test_concurrent_requests_during_spawn():
    """Verify concurrent requests wait for single spawn."""
    proxy = LazyHandlerProxy(
        model_cfg_dict={"model_path": "test", "model_id": "test", "model_type": "lm"},
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=300,
    )
    await proxy.initialize()

    # Mock spawn to take time
    spawn_event = asyncio.Event()
    async def slow_spawn(queue_config):
        await spawn_event.wait()
        # Create mock handler after event
        mock_handler = Mock()
        mock_handler.generate_text_response = AsyncMock(return_value={"response": "test"})
        proxy._handler = mock_handler

    proxy._ensure_handler = slow_spawn

    # Launch two concurrent requests
    results = await asyncio.gather([
        proxy.generate_text_response({"messages": ["req1"]}),
        proxy.generate_text_response({"messages": ["req2"]}),
    ])

    # Both should complete after spawn
    assert len(results) == 2
    assert all(r.get("response") == "test" for r in results)


@pytest.mark.asyncio
async def test_concurrent_during_unload():
    """Verify concurrent request cancels unload and spawns fresh."""
    proxy = LazyHandlerProxy(
        model_cfg_dict={"model_path": "test", "model_id": "test", "model_type": "lm"},
        model_type="lm",
        model_path="test",
        model_id="test",
        idle_timeout_seconds=1,  # Very short for testing
    )
    await proxy.initialize()

    # Set handler and simulate unload starting
    proxy._handler = Mock()
    proxy._unload_handler = AsyncMock()
    unload_started = asyncio.Event()

    async def mock_unload():
        unload_started.set()
        await asyncio.sleep(10)  # Hold during request

    proxy._unload_handler = mock_unload

    # Start unload in background
    unload_task = asyncio.create_task(mock_unload())
    await unload_started.wait()

    # Request arrives during unload
    mock_handler.generate_text_response = AsyncMock(return_value={"response": "new"})
    result = await proxy.generate_text_response({"messages": ["test"]})

    # Should use existing handler, not spawn new
    assert result == {"response": "new"}
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/core/test_lazy_handler.py::test_concurrent -v`
Expected: PASS

**Note:** Do NOT commit changes to source control.

---

## Task 11: End-to-end integration test

**Files:**
- Create: `tests/integration/test_lazy_e2e.py`

**Step 1: Write failing test for full lazy loading cycle**

```python
import pytest
from httpx import AsyncClient
from app.main import start_multi
from app.config import MultiModelServerConfig, ModelEntryConfig
import asyncio


@pytest.mark.asyncio
async def test_lazy_load_unload_cycle():
    """Test full cycle: unloaded → request → ready → idle → unloaded."""
    # This is a sketch - would need actual server startup
    # For now, document test structure

    config = MultiModelServerConfig(
        host="127.0.0.1",
        port=8001,
        default_lazy_load=True,
        models=[
            ModelEntryConfig(
                model_path="mlx-community/tiny-model",  # Use tiny model for testing
                model_id="tiny-lazy",
                model_type="lm",
                lazy_load=True,
                idle_timeout_seconds=5,  # 5 seconds for testing
            )
        ]
    )

    # 1. Start server
    # 2. Check /v1/models - should show "unloaded"
    # 3. Make request - should spawn handler
    # 4. Check /v1/models - should show "ready"
    # 5. Wait 6 seconds (idle timeout + margin)
    # 6. Check /v1/models - should show "unloaded"
    # 7. Make another request - should spawn again

    # For unit test scope, verify config structure
    assert config.models[0].lazy_load is True
    assert config.models[0].idle_timeout_seconds == 5
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/integration/test_lazy_e2e.py::test_lazy_load_unload_cycle -v`
Expected: PASS (config validation at minimum)

**Note:** Do NOT commit changes to source control.

---

## Task 12: Add __all__ exports

**Files:**
- Modify: `app/core/__init__.py` (if exists)
- Modify: `app/core/lazy_handler.py`

**Step 1: Add exports to lazy_handler.py**

```python
# At top of app/core/lazy_handler.py
__all__ = ["LazyHandlerProxy"]
```

**Step 2: Run tests to verify nothing breaks**

Run: `pytest tests/ -v`
Expected: All tests still pass

**Note:** Do NOT commit changes to source control.

---

## Task 13: Update CHANGELOG

**Files:**
- Modify: `CHANGELOG.md` (if exists)

**Step 1: Add lazy loading entry**

```markdown
## [Unreleased]

### Added
- **Lazy model loading**: Models can be configured to load on-demand with `lazy_load: true`
- **Auto-unload**: Models automatically unload after configurable `idle_timeout_seconds` of inactivity
- **Preload option**: `preload: true` forces eager loading for critical models
- **Model status**: `/v1/models` endpoint now returns `"status": "ready"` or `"status": "unloaded"`
- **Server-level defaults**: `default_lazy_load` and `default_idle_timeout_seconds` for global configuration

### Changed
- Multi-handler mode now supports mixed eager and lazy loading in same config

### Fixed
- N/A
```

**Note:** Do NOT commit changes to source control.

---

## Task 14: Final verification and cleanup

**Files:**
- None (runs all tests)

**Step 1: Run full test suite**

Run: `pytest tests/ -v --cov=app`
Expected: All tests pass, coverage report generated

**Step 2: Check for TODOs and debug prints**

Run: `grep -r "TODO\|FIXME\|print(" app/ tests/`
Expected: No TODOs or debug prints in production code

**Step 3: Verify example config is valid**

Run: `python -c "from app.config import load_config_from_yaml; load_config_from_yaml('examples/config.yaml')"`
Expected: No errors, config parsed correctly

**Note:** Development complete. Manual testing recommended before creating a pull request to upstream repository.

---

## Summary

This implementation adds lazy loading with 14 bite-sized tasks covering:

1. Configuration schema extensions with full backward compatibility
2. LazyHandlerProxy wrapper class with lifecycle management
3. Request forwarding through all handler methods
4. Server integration for mixed lazy/eager modes
5. Model status reporting via `/v1/models`
6. Example configuration documentation
7. README documentation section
8. Integration tests for config validation
9. Unit tests for core LazyHandlerProxy behavior
10. Concurrent request handling with locking
11. End-to-end test structure
12. Proper module exports
13. CHANGELOG updates
14. Final verification and cleanup

**Total estimated time:** 3-4 hours depending on testing depth
**Test coverage goal:** >90% for new code paths

**Upstream Repository:** https://github.com/piscopoc/mlx-openai-server
