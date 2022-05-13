from typing import Any, Dict

import httpx
from retrying import retry

from ...models.create_endpoint_parameters import CreateEndpointParameters
from ...models.endpoint import Endpoint
from ...types import Response
from ...util.errors import QCSHTTPStatusError, raise_for_status
from ...util.retry import DEFAULT_RETRY_ARGUMENTS


def _get_kwargs(
    *,
    json_body: CreateEndpointParameters,
) -> Dict[str, Any]:

    json_json_body = json_body.to_dict()

    return {
        "json": json_json_body,
    }


def _parse_response(*, response: httpx.Response) -> Endpoint:
    raise_for_status(response)
    if response.status_code == 201:
        response_201 = Endpoint.from_dict(response.json())

        return response_201
    else:
        raise QCSHTTPStatusError(
            f"Unexpected response: status code {response.status_code}", response=response, error=None
        )


def _build_response(*, response: httpx.Response) -> Response[Endpoint]:
    """
    Construct the Response class from the raw ``httpx.Response``.
    """
    return Response.build_from_httpx_response(response=response, parse_function=_parse_response)


@retry(**DEFAULT_RETRY_ARGUMENTS)
def sync(
    *,
    client: httpx.Client,
    json_body: CreateEndpointParameters,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[Endpoint]:
    url = "/v1/endpoints"

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
) -> Response[Endpoint]:
    json_body = CreateEndpointParameters.from_dict(json_body_dict)

    url = "/v1/endpoints"

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
    json_body: CreateEndpointParameters,
    httpx_request_kwargs: Dict[str, Any] = {},
) -> Response[Endpoint]:
    url = "/v1/endpoints"

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
) -> Response[Endpoint]:
    json_body = CreateEndpointParameters.from_dict(json_body_dict)

    url = "/v1/endpoints"

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
