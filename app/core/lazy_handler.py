import asyncio
import time
from typing import Any, AsyncGenerator
from loguru import logger

from app.core.handler_process import HandlerProcessProxy

__all__ = ["LazyHandlerProxy"]


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
        self._queue_config: dict[str, Any] = {}

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
        """Unload handler process with lock."""
        async with self._loading_lock:
            if self._handler is None:
                return

            logger.info(f"Idle timeout for model '{self.model_id}', unloading...")
            try:
                await self._handler.cleanup()
            except Exception as e:
                logger.error(f"Error unloading handler: {e}")
            finally:
                self._handler = None
                self._model_created = 0
                logger.info(f"Model '{self.model_id}' unloaded")

    async def _ensure_handler(self, queue_config: dict[str, Any] | None = None) -> HandlerProcessProxy:
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
            # Use stored queue_config if none provided
            qc = queue_config if queue_config is not None else self._queue_config
            await proxy.start(qc)
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
        if queue_config:
            self._queue_config = queue_config

    async def cleanup(self) -> None:
        """Clean up handler if loaded."""
        self._running = False
        if self._handler:
            await self._handler.cleanup()
            self._handler = None
        if self._unload_task and not self._unload_task.done():
            self._unload_task.cancel()

    @property
    def status(self) -> str:
        """Return current load status."""
        if self._handler is None:
            return "unloaded"
        return "ready"

    # Placeholder methods matching HandlerProcessProxy interface
    async def get_models(self) -> list[dict[str, Any]]:
        return [{"id": self.model_id, "object": "model"}]

    async def get_queue_stats(self) -> dict[str, Any]:
        return {"queue_size": 0, "active_requests": 0}

    async def generate_text_stream(
        self, request: Any
    ) -> AsyncGenerator[Any, None]:
        """Forward streaming text generation, spawning handler if needed."""
        handler = await self._ensure_handler()
        self._reset_idle_timer()
        self._start_idle_timer_if_needed()

        if hasattr(handler, "generate_text_stream"):
            async for chunk in handler.generate_text_stream(request):
                self._reset_idle_timer()  # Update on each chunk
                yield chunk
        else:
            raise AttributeError("Handler does not support generate_text_stream")

    async def generate_text_response(self, request: Any) -> dict[str, Any]:
        """Forward non-streaming text generation, spawning handler if needed."""
        handler = await self._ensure_handler()
        self._reset_idle_timer()
        self._start_idle_timer_if_needed()
        return await handler.generate_text_response(request)

    async def generate_multimodal_stream(
        self, request: Any
    ) -> AsyncGenerator[Any, None]:
        handler = await self._ensure_handler()
        self._reset_idle_timer()
        self._start_idle_timer_if_needed()
        async for chunk in handler.generate_multimodal_stream(request):
            self._reset_idle_timer()
            yield chunk

    async def generate_multimodal_response(self, request: Any) -> dict[str, Any]:
        handler = await self._ensure_handler()
        self._reset_idle_timer()
        self._start_idle_timer_if_needed()
        return await handler.generate_multimodal_response(request)

    async def generate_embeddings_response(self, request: Any) -> Any:
        handler = await self._ensure_handler()
        self._reset_idle_timer()
        self._start_idle_timer_if_needed()
        return await handler.generate_embeddings_response(request)

    async def generate_image(self, request: Any) -> Any:
        handler = await self._ensure_handler()
        self._reset_idle_timer()
        self._start_idle_timer_if_needed()
        return await handler.generate_image(request)

    async def edit_image(self, request: Any) -> Any:
        handler = await self._ensure_handler()
        self._reset_idle_timer()
        self._start_idle_timer_if_needed()
        return await handler.edit_image(request)

    async def prepare_transcription_request(self, request: Any) -> dict[str, Any]:
        handler = await self._ensure_handler()
        self._reset_idle_timer()
        self._start_idle_timer_if_needed()
        return await handler.prepare_transcription_request(request)

    async def generate_transcription_response(self, request: Any) -> Any:
        handler = await self._ensure_handler()
        self._reset_idle_timer()
        self._start_idle_timer_if_needed()
        return await handler.generate_transcription_response(request)

    async def generate_transcription_stream_from_data(
        self, request_data: dict[str, Any], *args: Any, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        handler = await self._ensure_handler()
        self._reset_idle_timer()
        self._start_idle_timer_if_needed()
        async for chunk in handler.generate_transcription_stream_from_data(
            request_data, *args, **kwargs
        ):
            self._reset_idle_timer()
            yield chunk

    async def generate_speech_response(self, request: Any) -> Any:
        handler = await self._ensure_handler()
        self._reset_idle_timer()
        self._start_idle_timer_if_needed()
        return await handler.generate_speech_response(request)
