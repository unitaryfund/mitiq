from typing import Any, Dict, Union

import httpx
from retrying import retry

from ...models.list_endpoints_response import ListEndpointsResponse
from ...types import UNSET, Response, Unset
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    *,
    filter: Union[Unset, str] = UNSET,
    page_size: Union[Unset, int] = 10,
    page_token: Union[Unset, str] = UNSET,
) -> Dict[str, Any]:

    params: Dict[str, Any] = {}
    if filter != UNSET:
        params["filter"] = filter
    if page_size != UNSET:
        params["pageSize"] = page_size
    if page_token != UNSET:
        params["pageToken"] = page_token

    return {
        "params": params,
    }


def _parse_response(*, response: httpx.Response) -> ListEndpointsResponse:
    raise_for_status(response)
    if response.status_code == 200:
        response_200 = ListEndpointsResponse.from_dict(response.json())

        return response_200
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[ListEndpointsResponse]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    *,
    client: httpx.Client,
    filter: Union[Unset, str] = UNSET,
    page_size: Union[Unset, int] = 10,
    page_token: Union[Unset, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListEndpointsResponse]:
    url = "/v1/endpoints"

    kwargs = _get_kwargs(
        filter=filter,
        page_size=page_size,
        page_token=page_token,
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
    filter: Union[Unset, str] = UNSET,
    page_size: Union[Unset, int] = 10,
    page_token: Union[Unset, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListEndpointsResponse]:

    url = "/v1/endpoints"

    kwargs = _get_kwargs(
        filter=filter,
        page_size=page_size,
        page_token=page_token,
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
    filter: Union[Unset, str] = UNSET,
    page_size: Union[Unset, int] = 10,
    page_token: Union[Unset, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListEndpointsResponse]:
    url = "/v1/endpoints"

    kwargs = _get_kwargs(
        filter=filter,
        page_size=page_size,
        page_token=page_token,
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
    filter: Union[Unset, str] = UNSET,
    page_size: Union[Unset, int] = 10,
    page_token: Union[Unset, str] = UNSET,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListEndpointsResponse]:

    url = "/v1/endpoints"

    kwargs = _get_kwargs(
        filter=filter,
        page_size=page_size,
        page_token=page_token,
    )
    kwargs.update(httpx_request_kwargs)
    response = await client.request(
        "get",
        url,
        **kwargs,
    )
    return _build_response(response=response)
