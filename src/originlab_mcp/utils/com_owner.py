"""Single owner thread for every Origin COM call.

Origin's automation objects (originpro / OriginExt) are apartment-bound: an
object obtained on one thread fails, or reports "object is not connected to the
server", when it is used from another thread. The server used to touch Origin
from a fresh worker thread per call (soft timeout), from FastMCP's handler
thread, and from ``threading.Timer`` threads (idle detach, autosave), so calls
alternated between working and failing and could crash the server process.

``ComOwner`` runs all of them on one long-lived thread instead. Calls made from
that thread itself run inline, so code already on the owner thread never queues
behind itself.

The worker is a plain daemon thread fed by a queue rather than a
``ThreadPoolExecutor``: executors refuse new work once the interpreter starts
shutting down, which would stop the atexit ``OriginManager.shutdown()`` from
detaching cleanly, and a non-daemon idle worker would block interpreter exit.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, TypeVar

T = TypeVar("T")

_Item = tuple[Callable[[], Any], "Future[Any]"]


class ComOwner:
    """Lazily started single thread that owns the Origin COM link."""

    def __init__(self, name: str = "origin-com") -> None:
        self._name = name
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue[_Item | None] | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _loop(work: queue.SimpleQueue[_Item | None]) -> None:
        while True:
            item = work.get()
            if item is None:
                return
            fn, future = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - delivered via result()
                future.set_exception(exc)

    def _ensure_started(self) -> queue.SimpleQueue[_Item | None]:
        with self._lock:
            if self._queue is None or self._thread is None or not self._thread.is_alive():
                work: queue.SimpleQueue[_Item | None] = queue.SimpleQueue()
                thread = threading.Thread(
                    target=self._loop, args=(work,), name=self._name, daemon=True
                )
                thread.start()
                self._queue, self._thread = work, thread
            return self._queue

    @property
    def thread_ident(self) -> int | None:
        """Ident of the owner thread once it has started (None before)."""
        thread = self._thread
        return thread.ident if thread is not None else None

    def on_owner_thread(self) -> bool:
        thread = self._thread
        return thread is not None and threading.current_thread() is thread

    def submit(self, fn: Callable[[], T]) -> Future[T]:
        """Queue ``fn`` on the owner thread (run inline if already on it)."""
        future: Future[T] = Future()
        if self.on_owner_thread():
            future.set_running_or_notify_cancel()
            try:
                future.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - delivered via result()
                future.set_exception(exc)
            return future
        self._ensure_started().put((fn, future))
        return future

    def call(self, fn: Callable[[], T], timeout: float | None = None) -> T:
        """Run ``fn`` on the owner thread and wait for its result.

        ``timeout`` raises ``concurrent.futures.TimeoutError`` without
        cancelling a call that has already started.
        """
        return self.submit(fn).result(timeout=timeout)

    def shutdown(self) -> None:
        """Let the owner thread exit after queued work. Never waits for it."""
        with self._lock:
            work, self._queue, self._thread = self._queue, None, None
        if work is not None:
            work.put(None)
