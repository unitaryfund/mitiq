import asyncio
import threading
import httpx
from pydantic import BaseModel, Field
from http import HTTPStatus
from typing import Set

from qcs_api_client.client._configuration import (
    QCSClientConfiguration,
    TokenPayload,
)


class QCSAuthConfiguration(BaseModel):
    pre: bool = False
    post: bool = True
    post_refresh_statuses: Set[int] = Field(default_factory=lambda: {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN})


class QCSAuthRefreshError(Exception):
    def __init__(self, response: httpx.Response):
        self.response = response
        self.message = f"authentication token refresh failed with status {response.status_code}: {response.text}"
        super().__init__(self.message)


class QCSAuth(httpx.Auth):
    def __init__(self, client_configuration: QCSClientConfiguration, auth_configuration: QCSAuthConfiguration = None):
        self._client_configuration = client_configuration
        self._sync_lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self._auth_configuration = auth_configuration or QCSAuthConfiguration()

    def sync_refresh_token(self):
        with self._sync_lock:
            refresh_token = self._client_configuration.credentials.refresh_token
            res = _do_refresh_token(
                self._client_configuration.auth_server.token_url(),
                self._client_configuration.auth_server.client_id,
                refresh_token,
            )
            if res.status_code != HTTPStatus.OK:
                raise QCSAuthRefreshError(response=res)
            token_payload = TokenPayload(**res.json())
            self._client_configuration.secrets.update_token(
                credentials_name=self._client_configuration.profile_name, token=token_payload
            )

    def sync_auth_flow(self, request):
        if self._client_configuration.credentials.token_payload is None:
            yield request
            return

        if self._auth_configuration.pre and self._client_configuration.credentials.token_payload.should_refresh():
            self.sync_refresh_token()

        request.headers["Authorization"] = f"Bearer {self._client_configuration.credentials.access_token}"
        response = yield request

        if self._auth_configuration.post and response.status_code in self._auth_configuration.post_refresh_statuses:
            self.sync_refresh_token()
            request.headers["Authorization"] = f"Bearer {self._client_configuration.credentials.access_token}"
            yield request

    async def async_refresh_token(self):
        async with self._async_lock:
            refresh_token = self._client_configuration.credentials.refresh_token
            res = await _do_refresh_token_async(
                self._client_configuration.auth_server.token_url(),
                self._client_configuration.auth_server.client_id,
                refresh_token,
            )
            if res.status_code != HTTPStatus.OK:
                raise QCSAuthRefreshError(response=res)
            token_payload = TokenPayload(**res.json())
            self._client_configuration.secrets.update_token(
                credentials_name=self._client_configuration.profile_name, token=token_payload
            )

    async def async_auth_flow(self, request):
        if self._client_configuration.credentials.token_payload is None:
            yield request
            return

        if self._auth_configuration.pre and self._client_configuration.credentials.token_payload.should_refresh():
            await self.async_refresh_token()

        request.headers["Authorization"] = f"Bearer {self._client_configuration.credentials.access_token}"
        response = yield request

        if self._auth_configuration.post and response.status_code in self._auth_configuration.post_refresh_statuses:
            await self.async_refresh_token()
            request.headers["Authorization"] = f"Bearer {self._client_configuration.credentials.access_token}"
            yield request


def _do_refresh_token(token_url: str, client_id: str, refresh_token: str):
    data = {"grant_type": "refresh_token", "client_id": client_id, "refresh_token": refresh_token}
    return httpx.request(
        "POST",
        token_url,
        data=data,
    )


async def _do_refresh_token_async(token_url: str, client_id: str, refresh_token: str):
    data = {"grant_type": "refresh_token", "client_id": client_id, "refresh_token": refresh_token}
    async with httpx.AsyncClient() as client:
        return await client.request(
            "POST",
            token_url,
            data=data,
        )
