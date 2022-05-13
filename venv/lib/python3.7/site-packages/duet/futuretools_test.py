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

from concurrent.futures import Future
from typing import Any, Callable, Optional

import pytest

import duet

try:
    import grpc
except ImportError:
    grpc = None


def test_awaitable_future():
    assert isinstance(duet.awaitable(Future()), duet.AwaitableFuture)


@pytest.mark.skipif(grpc is None, reason="only run if grpc is installed")
def test_awaitable_grpc_future():
    class ConcreteGrpcFuture(grpc.Future):
        def cancel(self) -> bool:
            return True

        def cancelled(self) -> bool:
            return True

        def running(self) -> bool:
            return True

        def done(self) -> bool:
            return True

        def result(self, timeout: Optional[int] = None) -> Any:
            return 1234

        def exception(self, timeout=None) -> Optional[BaseException]:
            return None

        def add_done_callback(self, fn: Callable[[Any], Any]) -> None:
            pass

        def traceback(self, timeout=None):
            pass

    assert isinstance(duet.awaitable(ConcreteGrpcFuture()), duet.AwaitableFuture)
