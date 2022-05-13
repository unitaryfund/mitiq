from typing import Any, Dict

import httpx
from retrying import retry

from ...models.list_client_applications_response import ListClientApplicationsResponse
from ...types import Response
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs() -> Dict[str, Any]:

    return {}


def _parse_response(*, response: httpx.Response) -> ListClientApplicationsResponse:
    raise_for_status(response)
    if response.status_code == 200:
        response_200 = ListClientApplicationsResponse.from_dict(response.json())

        return response_200
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[ListClientApplicationsResponse]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    *,
    client: httpx.Client,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListClientApplicationsResponse]:
    url = "/v1/clientApplications"

    kwargs = _get_kwargs()
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
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListClientApplicationsResponse]:

    url = "/v1/clientApplications"

    kwargs = _get_kwargs()
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
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListClientApplicationsResponse]:
    url = "/v1/clientApplications"

    kwargs = _get_kwargs()
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
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[ListClientApplicationsResponse]:

    url = "/v1/clientApplications"

    kwargs = _get_kwargs()
    kwargs.update(httpx_request_kwargs)
    response = await client.request(
        "get",
        url,
        **kwargs,
    )
    return _build_response(response=response)
