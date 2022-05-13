from httpx import Response, HTTPStatusError
from http import HTTPStatus
from typing import cast, Dict, Any, Optional
from json import JSONDecodeError

from ..models.error import Error


class QCSHTTPStatusError(HTTPStatusError):
    def __init__(self, message: str, *, response: Response, error: Optional[Error]) -> None:
        super().__init__(message, request=response.request, response=response)
        self.error = error


def raise_for_status(res: Response):
    """
    Raise the `QCSHTTPStatusError` if one occurred.
    """
    if res.request is None:
        raise RuntimeError(
            "Cannot call `raise_for_status` as the request " "instance has not been set on this response."
        )
    elif res.status_code < HTTPStatus.BAD_REQUEST:
        return None

    message = f"QCS API call {res.request.method} {res.request.url} failed with status {res.status_code}: {res.text}"
    error = None
    try:
        error = Error.from_dict(cast(Dict[str, Any], res.json()))
    except (JSONDecodeError, KeyError):
        pass

    raise QCSHTTPStatusError(message, error=error, response=res)
