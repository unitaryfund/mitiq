##############################################################################
# Copyright 2018 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
"""Utils for message passing"""
import uuid
import warnings
from inspect import signature
from typing import Callable, Optional, Tuple, Union, List, Any

import rpcq.messages


def rpc_warning(warning: warnings.WarningMessage) -> rpcq.messages.RPCWarning:
    return rpcq.messages.RPCWarning(body=str(warning.message),
                                    kind=str(warning.category.__name__))

def rpc_request(method_name: str, *args, **kwargs) -> rpcq.messages.RPCRequest:
    """
    Create RPC request

    :param method_name: Method name
    :param args: Positional arguments
    :param kwargs: Keyword arguments
    :return: JSON RPC formatted dict
    """
    if args:
        kwargs['*args'] = args

    return rpcq.messages.RPCRequest(
        jsonrpc='2.0',
        id=str(uuid.uuid4()),
        method=method_name,
        params=kwargs
    )


def rpc_reply(id: Union[str, int], result: Optional[object],
              warnings: Optional[List[Warning]] = None) -> rpcq.messages.RPCReply:
    """
    Create RPC reply

    :param str|int id: Request ID
    :param result: Result
    :param warnings: List of warnings to attach to the message
    :return: JSON RPC formatted dict
    """
    warnings = warnings or []

    return rpcq.messages.RPCReply(
        jsonrpc='2.0',
        id=id,
        result=result,
        warnings=[rpc_warning(warning) for warning in warnings]
    )


def rpc_error(id: Union[str, int], error_msg: str,
              warnings: List[Any] = []) -> rpcq.messages.RPCError:
    """
    Create RPC error

    :param id: Request ID
    :param error_msg: Error message
    :param warning: List of warnings to attach to the message
    :return: JSON RPC formatted dict
    """
    return rpcq.messages.RPCError(
        jsonrpc='2.0',
        id=id,
        error=error_msg,
        warnings=[rpc_warning(warning) for warning in warnings])


def get_input(params: Union[dict, list]) -> Tuple[list, dict]:
    """
    Get positional or keyword arguments from JSON RPC params

    :param params: Parameters passed through JSON RPC
    :return: args, kwargs
    """
    # Backwards compatibility for old clients that send params as a list
    if isinstance(params, list):
        args = params
        kwargs = {}
    elif isinstance(params, dict):
        args = params.pop('*args', [])
        kwargs = params
    else:  # pragma no coverage
        raise TypeError(
            'Unknown type {} of params, must be list or dict'.format(type(params)))

    return args, kwargs


def get_safe_input(params: Union[dict, list], handler: Callable) -> Tuple[list, dict]:
    """
    Get positional or keyword arguments from JSON RPC params,
       filtering out kwargs that aren't in the function signature

    :param params: Parameters passed through JSON RPC
    :param handler: RPC handler function
    :return: args, kwargs
    """
    args, kwargs = get_input(params)

    handler_signature = signature(handler)
    kwargs = { k: v for k, v in kwargs.items() if k in handler_signature.parameters }

    return args, kwargs


class RPCErrorError(IOError):
    """JSON RPC error that is raised by a Client when it receives an RPCError message"""

    def __init__(self, *args, **kwargs):
        if type(self) is RPCErrorError:
            warnings.warn("`RPCErrorError` is deprecated in favor of the "
                          "less-loquacious `RPCError`.", DeprecationWarning)

        super().__init__(*args, **kwargs)


class RPCError(RPCErrorError):
    """JSON RPC error that is raised by a Client when it receives an RPCError message"""


class RPCMethodError(AttributeError):
    """JSON RPC error that is raised by JSON RPC spec for nonexistent methods"""


class catch_warnings(warnings.catch_warnings):
    """This variant of warnings.catch_warnings both logs *and* re-emits warnings."""
    def __enter__(self):
        super().__enter__()

        # the super() method does most of the work.  what follows below is actually
        # also cut out of the super() method, but the relevant line there is
        #
        #     self._module._showwarnmsg_impl = log.append
        #
        # we, on the other hand, want to both append *and* call the saved parent
        # log-displayer, so we wrap both inside of new_logger and store that instead.

        if self._record:
            log = []
            def new_logger(msg):
                nonlocal log, self
                log.append(msg)
                self._showwarnmsg_impl(msg)
            self._module._showwarnmsg_impl = new_logger
            return log
        else:
            return None
