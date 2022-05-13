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

"""Run asynchronous coroutines using Futures.

Coroutines using async/await provide a way to write computations that can be
paused and later resumed. This module provides a way to manage the execution of
multiple such coroutines using Futures to provide concurrency. In other words,
while one coroutine is waiting for a particular Future to complete, other
coroutines can run.

Other libraries for dealing with async/await, such as asyncio in the standard
library or the third-party trio library, are focused on providing fully
asynchronous I/O capabilities. Here we focus solely on managing coroutines and
rely on Futures (themselves backed by either threads or a separate async I/O
library) to provide concurrency. This module differs from those other libraries
in two big ways: first, it is reentrant, meaning we can call `duet.run`
recursively, which makes it much easier to refactor our code incrementally to
be asynchronous; second, we can run the event loop manually one tick at a time,
which makes it possible to implement things like the pmap function below which
wraps async code into a generator interface.
"""

from concurrent.futures import CancelledError

from duet._version import __version__
from duet.aitertools import aenumerate, aiter, AnyIterable, AsyncCollector, azip
from duet.api import (
    awaitable,
    awaitable_func,
    deadline_scope,
    LimitedScope,
    Limiter,
    new_scope,
    pmap,
    pmap_aiter,
    pmap_async,
    pstarmap,
    pstarmap_aiter,
    pstarmap_async,
    run,
    Scope,
    sleep,
    sync,
    timeout_scope,
)
from duet.futuretools import AwaitableFuture, BufferedFuture, completed_future, failed_future
