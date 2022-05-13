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
"""
Server that accepts JSON RPC requests and returns JSON RPC replies/errors.
"""
import asyncio
import logging
import sys
from asyncio import AbstractEventLoop
from typing import Callable, List, Optional, Tuple
from datetime import datetime

import zmq.asyncio
from zmq.auth.asyncio import AsyncioAuthenticator

from rpcq._base import to_msgpack, from_msgpack
from rpcq._spec import RPCSpec
from rpcq.messages import RPCRequest

if sys.version_info < (3, 7):
    from rpcq.external.dataclasses import dataclass
else:
    from dataclasses import dataclass

_log = logging.getLogger(__name__)

# Required values for ZeroMQ curve authentication, in lieu of a TypedDict
@dataclass
class ServerAuthConfig:
    server_secret_key: bytes
    server_public_key: bytes
    client_keys_directory: str = ""
        

class Server:
    """
    Server that accepts JSON RPC calls through a socket.
    """
    def __init__(self, rpc_spec: RPCSpec = None, announce_timing: bool = False,
                 serialize_exceptions: bool = True, auth_config: Optional[ServerAuthConfig] = None):
        """
        Create a server that will be linked to a socket

        :param rpc_spec: JSON RPC spec
        :param announce_timing:
        :param serialize_exceptions: If set to True, this Server will catch all exceptions occurring
            internally to it and, when possible, communicate them to the interrogating Client.  If
            set to False, this Server will re-raise any exceptions it encounters (including, but not
            limited to, those which might occur through method calls to rpc_spec) for Server's
            local owner to handle.

            IMPORTANT NOTE: When set to False, this *almost definitely* means an unrecoverable
            crash, and the Server should then be _shutdown().
        :param auth_config: The configuration values necessary to enable Curve ZeroMQ authentication.
            These must be provided at instantiation, so they are available between the creation of the 
            context and socket.
        """
        self.announce_timing = announce_timing
        self.serialize_exceptions = serialize_exceptions

        self.rpc_spec = rpc_spec if rpc_spec else RPCSpec(serialize_exceptions=serialize_exceptions)
        self._exit_handlers = []

        self._socket = None
        self._auth_config = auth_config
        self._authenticator = None
        self._preloaded_keys = None

    def rpc_handler(self, f: Callable):
        """
        Add a function to the server. It will respond to JSON RPC requests with the corresponding method name.
        This can be used as both a side-effecting function or as a decorator.

        :param f: Function to add
        :return: Function wrapper (so it can be used as a decorator)
        """
        return self.rpc_spec.add_handler(f)

    def exit_handler(self, f: Callable):
        """
        Add an exit handler - a function which will be called when the server shuts down.

        :param f: Function to add
        """
        self._exit_handlers.append(f)

    async def recv_multipart(self):
        if self.auth_enabled:
            return await self.recv_multipart_with_auth()
        else:
            # If auth is not enabled, then the client "User-Id" will not be retrieved from
            #   the frames received, and we return None for that value.
            return (*await self._socket.recv_multipart(), None)

    async def recv_multipart_with_auth(self) -> Tuple[bytes, list, bytes]:
        """
        Code taken from pyzmq itself: https://github.com/zeromq/pyzmq/blob/master/zmq/sugar/socket.py#L449
          and then adapted to allow us to access the information in the frames.

        When copy=True, only the contents of the messages are returned, rather than the messages themselves.
          The message is necessary to be able to fetch the "User-Id", which is the public key the client used
          to connect to this socket.

        When using auth, knowing which client sent which message is important for authentication, and so 
          we reimplement recv_multipart here, and return the client key as the final member of a tuple
        """

        copy = False
        # Given a ROUTER socket, the first frame will be the sender's identity. 
        #   While, per the docs, this _should_ be retrievable from any frame with
        #   frame.get('Identity'), in practice this value was always blank.
        #   If we don't record the identity value, messages cannot be returned to
        #   the correct client.
        identity_frame = await self._socket.recv(0, copy=copy, track=False)
        identity = identity_frame.bytes

        # The client_id is the public key the client used to establish this connection
        #   It can be retrieved from all frames after the first. Here, we assume it
        #   is the same among all frames, and set it to the value from the first frame
        client_key = None

        # After the identity frame, we assemble all further frames in a single buffer.
        parts = bytearray(b'')
        while self._socket.getsockopt(zmq.RCVMORE):
            part = await self._socket.recv(0, copy=copy, track=False)
            data = part.bytes
            if client_key is None:
                client_key = part.get('User-Id')
                if not isinstance(client_key, bytes) and client_key is not None:
                    client_key = client_key.encode('utf-8')
            parts += data

        _log.debug(f'Received authenticated request from client_key {client_key}')
    
        return (identity, parts, client_key)

    async def run_async(self, endpoint: str):
        """
        Run server main task (asynchronously).

        :param endpoint: Socket endpoint to listen to, e.g. "tcp://*:1234"
        """
        self._connect(endpoint)

        # spawn an initial listen task
        listen_task = asyncio.ensure_future(self.recv_multipart())
        task_list = [listen_task]

        while True:
            dones, pendings = await asyncio.wait(task_list, return_when=asyncio.FIRST_COMPLETED)

            # grab one "done" task to handle
            task_list, done_list = list(pendings), list(dones)
            done = done_list.pop()
            task_list += done_list

            if done == listen_task:
                try:
                    # empty_frame may either be:
                    # 1. a single null frame if the client is a REQ socket
                    # 2. an empty list (ie. no frames) if the client is a DEALER socket
                    identity, *empty_frame, msg, client_key = done.result()
                    request = from_msgpack(msg)
                    request.params['client_key'] = client_key

                    # spawn a processing task
                    task_list.append(asyncio.ensure_future(
                        self._process_request(identity, empty_frame, request)))
                except Exception as e:
                    if self.serialize_exceptions:
                        _log.exception('Exception thrown in Server run loop during request '
                                       'reception: {}'.format(repr(e)))
                    else:
                        raise e
                finally:
                    # spawn a new listen task
                    listen_task = asyncio.ensure_future(self.recv_multipart())
                    task_list.append(listen_task)
            else:
                # if there's been an exception during processing, consider reraising it
                try:
                    done.result()
                except Exception as e:
                    if self.serialize_exceptions:
                        _log.exception('Exception thrown in Server run loop during request '
                                       'dispatch: {}'.format(repr(e)))
                    else:
                        raise e

    def run(self, endpoint: str, loop: AbstractEventLoop = None):
        """
        Run server main task.

        :param endpoint: Socket endpoint to listen to, e.g. "tcp://*:1234"
        :param loop: Event loop to run server in (alternatively just use run_async method)
        """
        if not loop:
            loop = asyncio.get_event_loop()

        try:
            loop.run_until_complete(self.run_async(endpoint))
        except KeyboardInterrupt:
            self._shutdown()

    def stop(self):
        """
        DEPRECATED
        """
        pass

    def _shutdown(self):
        """
        Shut down the server.
        """
        for exit_handler in self._exit_handlers:
            exit_handler()

        if self._socket:
            self._socket.close()
            self._socket = None

    def _connect(self, endpoint: str):
        """
        Connect the server to an endpoint. Creates a ZMQ ROUTER socket for the given endpoint.

        :param endpoint: Socket endpoint, e.g. "tcp://*:1234"
        """
        if self._socket:
            raise RuntimeError('Cannot run multiple Servers on the same socket')

        context = zmq.asyncio.Context()
        self._socket = context.socket(zmq.ROUTER)
        self.start_auth(context)
        self._socket.bind(endpoint)

        _log.info("Starting server, listening on endpoint {}".format(endpoint))

    async def _process_request(self, identity: bytes, empty_frame: list, request: RPCRequest):
        """
        Executes the method specified in a JSON RPC request and then sends the reply to the socket.

        :param identity: Client identity provided by ZeroMQ
        :param empty_frame: Either an empty list or a single null frame depending on the client type
        :param request: JSON RPC request
        """
        try:
            _log.debug("Client %s sent request: %s", identity, request)
            start_time = datetime.now()
            reply = await self.rpc_spec.run_handler(request)
            if self.announce_timing:
                _log.info("Request {} for {} lasted {} seconds".format(
                    request.id, request.method, (datetime.now() - start_time).total_seconds()))

            _log.debug("Sending client %s reply: %s", identity, reply)
            await self._socket.send_multipart([identity, *empty_frame, to_msgpack(reply)])
        except Exception as e:
            if self.serialize_exceptions:
                _log.exception('Exception thrown in _process_request')
            else:
                raise e

    @property
    def auth_configured(self) -> bool:
        return (self._auth_config is not None) and isinstance(self._auth_config.server_secret_key, bytes) and isinstance(self._auth_config.server_public_key, bytes)

    @property
    def auth_enabled(self) -> bool:
        return bool(self._socket and self._socket.curve_server)

    def start_auth(self, context: zmq.Context) -> bool:
        """
        Starts the ZMQ auth service thread, enabling authorization on all sockets within this context
        """
        if not self.auth_configured:
            return False
        self._socket.curve_secretkey = self._auth_config.server_secret_key
        self._socket.curve_publickey = self._auth_config.server_public_key
        self._socket.curve_server = True
        self._authenticator = AsyncioAuthenticator(context)
        if self._preloaded_keys:
            self.set_client_keys(self._preloaded_keys)
        else:
            self.load_client_keys_from_directory()
        self._authenticator.start()
        return True

    def stop_auth(self) -> bool:
        """
        Stops the ZMQ auth service thread, allowing NULL authenticated clients (only) to connect to
            all threads within its context
        """
        if self._authenticator:
            self._socket.curve_server = False
            self._authenticator.stop()
            return True
        else:
            return False

    def load_client_keys_from_directory(self, directory: Optional[str] = None) -> bool:
        """
        Reset authorized public key list to those present in the specified directory
        """

        # The directory must either be specified at class creation or on each method call
        if directory is None:
            if self._auth_config.client_keys_directory:
                directory = self._auth_config.client_keys_directory
        if not directory or not self.auth_configured:
            return False
        self._authenticator.configure_curve(domain='*', location=directory)
        return True

    def set_client_keys(self, client_keys: List[bytes]):
        """
        Reset authorized public key list to this set. Avoids the disk read required by configure_curve,
            and allows keys to be managed externally.

        In some cases, keys may be preloaded before the authenticator is started. In this case, we 
            cache those preloaded keys
        """
        if self._authenticator:
            _log.debug(f"Authorizer: Setting client keys to {client_keys}")
            self._authenticator.certs['*'] = {key: True for key in client_keys}
        else:
            _log.debug(f"Authorizer: Preloading client keys to {client_keys}")
            self._preloaded_keys = client_keys
