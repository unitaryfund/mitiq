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

"""Mypy plugin to provide better typechecking of duet functions.

For more information about mypy plugins see:
https://mypy.readthedocs.io/en/stable/extending_mypy.html#extending-mypy-using-plugins
"""

from typing import Callable, Optional

from mypy.plugin import FunctionContext, Plugin
from mypy.types import CallableType, get_proper_type, Instance, Type


def duet_sync_callback(ctx: FunctionContext) -> Type:
    """Callback to provide an accurate signature for duet.sync.

    The duet.sync function wraps an async callable in a synchronous wrapper:

        def sync(f: Callable[..., Awaitable[T]]) -> Callable[..., T]:

    This plugin basically tells mypy that the two ellipses are exactly the same,
    that is, that the new synchronous callable accepts exactly the same args as
    the original function. This allows for precise typechecking of calls to
    functions wrapped by duet.sync.
    """
    func_type = get_proper_type(ctx.arg_types[0][0])
    if not isinstance(func_type, CallableType):
        ctx.api.msg.fail(f"expected Callable[..., Awaitable[T]], got {func_type}", ctx.context)
        return ctx.default_return_type

    # Note that the return type of an async function is Coroutine[Any, Any, T],
    # which is a subtype of Awaitable[T]. See:
    # https://mypy.readthedocs.io/en/stable/more_types.html#typing-async-await
    ret_type = get_proper_type(func_type.ret_type)
    if not (isinstance(ret_type, Instance) and ret_type.type.name == "Coroutine"):
        if not func_type.implicit:
            ctx.api.msg.fail(f"expected return type Awaitable[T], got {ret_type}", ctx.context)
        return ctx.default_return_type

    result_type = ret_type.args[-1]
    return func_type.copy_modified(ret_type=result_type)


class DuetPlugin(Plugin):
    def get_function_hook(self, fullname: str) -> Optional[Callable[[FunctionContext], Type]]:
        if fullname == "duet.api.sync":
            return duet_sync_callback
        return None


def plugin(version: str):
    return DuetPlugin
