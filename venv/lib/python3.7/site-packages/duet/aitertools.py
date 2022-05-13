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

import collections
from typing import (
    AsyncIterable,
    AsyncIterator,
    Deque,
    Generic,
    Iterable,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import duet.futuretools as futuretools

T = TypeVar("T")

AnyIterable = Union[Iterable[T], AsyncIterable[T]]


async def aenumerate(iterable: AnyIterable[T], start: int = 0) -> AsyncIterator[Tuple[int, T]]:
    i = start
    async for value in aiter(iterable):
        yield (i, value)
        i += 1


async def aiter(iterable: AnyIterable[T]) -> AsyncIterator[T]:
    if isinstance(iterable, Iterable):
        for value in iterable:
            yield value
    else:
        async for value in iterable:
            yield value


async def azip(*iterables: AnyIterable) -> AsyncIterator[Tuple]:
    iters = [aiter(iterable) for iterable in iterables]
    while True:
        values = []
        for it in iters:
            try:
                value = await it.__anext__()
                values.append(value)
            except StopAsyncIteration:
                return
        yield tuple(values)


class AsyncCollector(Generic[T]):
    """Allows async iteration over values dynamically added by the client.

    This class is useful for creating an asynchronous iterator that is "fed" by
    one process (the "producer") and iterated over by another process (the
    "consumer"). The producer calls `.add` repeatedly to add values to be
    iterated over, and then calls either `.done` or `.error` to stop the
    iteration or raise an error, respectively. The consumer can use `async for`
    or direct calls to `__anext__` to iterate over the produced values.
    """

    def __init__(self):
        self._buffer: Deque[T] = collections.deque()
        self._waiter: Optional[futuretools.AwaitableFuture[None]] = None
        self._done: bool = False
        self._error: Optional[Exception] = None

    def add(self, value: T) -> None:
        if self._done:
            raise RuntimeError("already done.")
        self._buffer.append(value)
        if self._waiter:
            self._waiter.try_set_result(None)

    def done(self) -> None:
        if self._done:
            raise RuntimeError("already done.")
        self._done = True
        if self._waiter:
            self._waiter.try_set_result(None)

    def error(self, error: Exception) -> None:
        if self._done:
            raise RuntimeError("already done.")
        self._done = True
        self._error = error
        if self._waiter:
            self._waiter.try_set_result(None)

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        if not self._done and not self._buffer:
            self._waiter = futuretools.AwaitableFuture()
            await self._waiter
            self._waiter = None
        if self._buffer:
            return self._buffer.popleft()
        if self._error:
            raise self._error
        raise StopAsyncIteration()
