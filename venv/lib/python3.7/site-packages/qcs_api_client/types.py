""" Contains some shared types for properties """
from typing import BinaryIO, Callable, Generic, Optional, TextIO, Tuple, TypeVar, Union

import attr
import httpx


class Unset:
    def __bool__(self) -> bool:
        return False


UNSET: Unset = Unset()


@attr.s(auto_attribs=True)
class File:
    """ Contains information for file uploads """

    payload: Union[BinaryIO, TextIO]
    file_name: Optional[str] = None
    mime_type: Optional[str] = None

    def to_tuple(self) -> Tuple[Optional[str], Union[BinaryIO, TextIO], Optional[str]]:
        """ Return a tuple representation that httpx will accept for multipart/form-data """
        return self.file_name, self.payload, self.mime_type


T = TypeVar("T")


class Response(httpx.Response, Generic[T]):
    """
    Response from an API endpoint.

    Serves as a minimal wrapper around an httpx.Response object, supporting parsing into a
    known API response type.
    """

    _parsed: Optional[T]
    """The response body parsed into an API model instance."""

    _parse_function: Callable[[httpx.Response], T]
    """The function responsible for parsing the response body into an API model instance."""

    @classmethod
    def build_from_httpx_response(
        cls, *, response: httpx.Response, parse_function: Callable[[httpx.Response], T]
    ) -> "Response":
        """
        Mutate an ``httpx.Response`` into a ``Response`` by respecifying the class and applying
        required values.
        """
        response.__class__ = cls
        response._parsed = None
        response._parse_function = parse_function
        return response

    @property
    def parsed(self) -> T:
        """
        Return the response body parsed into an API model.

        Value is memoized after the first successful call.
        """
        if self._parsed is None:
            self._parsed = self._parse_function(response=self)

        return self._parsed


__all__ = ["File", "Response"]
