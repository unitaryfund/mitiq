from typing import Any, Callable, Dict, List, Optional

import attr

from ..types import UNSET
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class EngagementCredentials:
    """Credentials are the ZeroMQ CURVE Keys used to encrypt the connection with the Quantum Processor
    Endpoint."""

    client_public: str
    client_secret: str
    server_public: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        client_public = self.client_public
        client_secret = self.client_secret
        server_public = self.server_public

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "clientPublic": client_public,
                "clientSecret": client_secret,
                "serverPublic": server_public,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "EngagementCredentials":
        d = src_dict.copy()
        client_public = d.pop("clientPublic")

        client_secret = d.pop("clientSecret")

        server_public = d.pop("serverPublic")

        engagement_credentials = EngagementCredentials(
            client_public=client_public,
            client_secret=client_secret,
            server_public=server_public,
        )

        engagement_credentials.additional_properties = d
        return engagement_credentials

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
