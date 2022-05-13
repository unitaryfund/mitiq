# Copyright 2021 The Duet Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
from concurrent.futures import Future
from typing import Any, Callable, Generator, Generic, Optional, Tuple, Type, TypeVar

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore[misc]

try:
    import grpc

    FutureClasses: Tuple[Type, ...] = (Future, grpc.Future)
except ImportError:
    FutureClasses = (Future,)


T = TypeVar("T")


class FutureLike(Protocol[T]):
    def result(self) -> T:
        ...

    def exception(self) -> Optional[BaseException]:
        ...

    def add_done_callback(self, fn: Callable[["FutureLike[T]"], Any]) -> None:
        ...


class AwaitableFuture(Future, Generic[T]):
    """A Future that can be awaited."""

    # This is an internal variable in the Future class.
    # We add an annotation here so mypy will let us use it.
    _condition: threading.Condition

    @staticmethod
    def isfuture(value: Any) -> bool:
        return isinstance(value, FutureClasses)

    @staticmethod
    def wrap(future: FutureLike[T]) -> "AwaitableFuture[T]":
        """Creates an awaitable future that wraps the given source future."""
        awaitable = AwaitableFuture[T]()

        def callback(future: FutureLike[T]):
            error = future.exception()
            if error is None:
                awaitable.try_set_result(future.result())
            else:
                awaitable.try_set_exception(error)

        future.add_done_callback(callback)
        return awaitable

    def __await__(self) -> Generator["AwaitableFuture[T]", None, T]:
        yield self
        return self.result()

    def try_set_result(self, result: T) -> bool:
        """Sets the result on this future if not already done.

        Returns:
            True if we set the result, False if the future was already done.
        """
        with self._condition:
            if self.done():
                return False
            self.set_result(result)
            return True

    def try_set_exception(self, exception: Optional[BaseException]) -> bool:
        """Sets an exception on this future if not already done.

        Returns:
            True if we set the exception, False if the future was already done.
        """
        with self._condition:
            if self.done():
                return False
            self.set_exception(exception)
            return True


class BufferedFuture(AwaitableFuture):
    """A future whose async operation may be buffered until flush is called.

    Calling the flush method starts the asynchronous operation associated with
    this future, if it has not been started already. By default, calling
    result or exception will also call flush so that the async operation will
    start and we do not deadlock waiting for a result.
    """

    def flush(self):
        pass

    def result(self, timeout=None):
        self.flush()
        return super().result(timeout)

    def exception(self, timeout=None):
        self.flush()
        return super().exception(timeout)


class BufferGroup:
    """A group of buffered futures that need to be flushed."""

    def __init__(self, latch=False):
        """

        Args:
            latch: If True, we set a flag the first time the group is flushed;
                we then immediately flush any futures added after that point.
                If False, the default, we store all added futures in a list and
                flush them the next time the group is flushed, regardless of
                whether the group has been flushed before.
        """
        self._latch = latch
        self._flushed = False
        self._futures = []

    def add(self, future):
        if not isinstance(future, BufferedFuture):
            return
        if self._latch and self._flushed:
            future.flush()
        else:
            self._futures.append(future)

    def flush(self):
        for f in self._futures:
            f.flush()
        self._futures.clear()
        if self._latch:
            self._flushed = True


class FutureList(BufferedFuture):
    """A Future that waits for a list of other Futures."""

    def __init__(self, futures):
        super().__init__()
        if not len(futures):
            self.set_result([])
            return
        self._results = [None] * len(futures)
        self._outstanding = len(futures)
        self._lock = threading.Lock()
        self._buffer = BufferGroup()
        for i, f in enumerate(futures):
            self._buffer.add(f)
            f.add_done_callback(lambda f, idx=i: self._handle_result(f, idx))

    def _handle_result(self, future, index):
        if self.done():
            return
        error = future.exception()
        if error is not None:
            self.try_set_exception(error)
            return
        result = future.result()
        with self._lock:
            self._results[index] = result
            self._outstanding -= 1
            if not self._outstanding:
                self.try_set_result(self._results)

    def flush(self):
        self._buffer.flush()


def completed_future(data: T) -> AwaitableFuture[T]:
    """Return a future with the given data as its result."""
    f = AwaitableFuture[T]()
    f.set_result(data)
    return f


def failed_future(error: BaseException) -> AwaitableFuture[Any]:
    """Return a future that will fail with the given error."""
    f = AwaitableFuture[Any]()
    f.set_exception(error)
    return f
