from typing import Any, Dict

import httpx
from retrying import retry

from ...models.add_group_user_request import AddGroupUserRequest
from ...types import Response
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    *,
    json_body: AddGroupUserRequest,
) -> Dict[str, Any]:

    json_json_body = json_body.to_dict()

    return {
        "json": json_json_body,
    }


def _parse_response(*, response: httpx.Response) -> None:
    raise_for_status(response)
    if response.status_code == 204:
        response_204 = None

        return response_204
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[None]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    *,
    client: httpx.Client,
    json_body: AddGroupUserRequest,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[None]:
    url = "/v1/groups:addUser"

    kwargs = _get_kwargs(
        json_body=json_body,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        "post",
        url,
        **kwargs,
    )
    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync_from_dict(
    *,
    client: httpx.Client,
    json_body_dict: Dict,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[None]:
    json_body = AddGroupUserRequest.from_dict(json_body_dict)

    url = "/v1/groups:addUser"

    kwargs = _get_kwargs(
        json_body=json_body,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        "post",
        url,
        **kwargs,
    )
    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio(
    *,
    client: httpx.AsyncClient,
    json_body: AddGroupUserRequest,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[None]:
    url = "/v1/groups:addUser"

    kwargs = _get_kwargs(
        json_body=json_body,
    )
    kwargs.update(httpx_request_kwargs)
    response = await client.request(
        "post",
        url,
        **kwargs,
    )
    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio_from_dict(
    *,
    client: httpx.AsyncClient,
    json_body_dict: Dict,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[None]:
    json_body = AddGroupUserRequest.from_dict(json_body_dict)

    url = "/v1/groups:addUser"

    kwargs = _get_kwargs(
        json_body=json_body,
    )
    kwargs.update(httpx_request_kwargs)
    response = await client.request(
        "post",
        url,
        **kwargs,
    )
    return _build_response(response=response)
