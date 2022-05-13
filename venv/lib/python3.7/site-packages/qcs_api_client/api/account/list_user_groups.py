from typing import Any, Dict

import httpx
from retrying import retry

from ...models.list_groups_response import ListGroupsResponse
from ...types import Response
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    *,
    user_id: str,
) -> Dict[str, Any]:

    return {}


def _parse_response(*, response: httpx.Response) -> ListGroupsResponse:
    raise_for_status(response)
    if response.status_code == 200:
        response_200 = ListGroupsResponse.from_dict(response.json())

        return response_200
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[ListGroupsResponse]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    *,
    client: httpx.Client,
    user_id: str,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListGroupsResponse]:
    url = "/v1/users/{userId}/groups".format(
        userId=user_id,
    )

    kwargs = _get_kwargs(
        user_id=user_id,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        "get",
        url,
        **kwargs,
    )
    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync_from_dict(
    *,
    client: httpx.Client,
    user_id: str,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListGroupsResponse]:

    url = "/v1/users/{userId}/groups".format(
        userId=user_id,
    )

    kwargs = _get_kwargs(
        user_id=user_id,
    )
    kwargs.update(httpx_request_kwargs)
    response = client.request(
        "get",
        url,
        **kwargs,
    )
    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio(
    *,
    client: httpx.AsyncClient,
    user_id: str,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListGroupsResponse]:
    url = "/v1/users/{userId}/groups".format(
        userId=user_id,
    )

    kwargs = _get_kwargs(
        user_id=user_id,
    )
    kwargs.update(httpx_request_kwargs)
    response = await client.request(
        "get",
        url,
        **kwargs,
    )
    return _build_response(response=response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
async def asyncio_from_dict(
    *,
    client: httpx.AsyncClient,
    user_id: str,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListGroupsResponse]:

    url = "/v1/users/{userId}/groups".format(
        userId=user_id,
    )

    kwargs = _get_kwargs(
        user_id=user_id,
    )
    kwargs.update(httpx_request_kwargs)
    response = await client.request(
        "get",
        url,
        **kwargs,
    )
    return _build_response(response=response)
