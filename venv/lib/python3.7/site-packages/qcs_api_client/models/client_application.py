from typing import Any, Callable, Dict, List, Optional, Union

import attr

from ..models.client_applications_download_link import ClientApplicationsDownloadLink
from ..types import UNSET, Unset
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class ClientApplication:
    """  """

    latest_version: str
    name: str
    supported: bool
    details_uri: Union[Unset, str] = UNSET
    links: Union[Unset, List[ClientApplicationsDownloadLink]] = UNSET
    minimum_version: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        latest_version = self.latest_version
        name = self.name
        supported = self.supported
        details_uri = self.details_uri
        links: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.links, Unset):
            links = []
            for links_item_data in self.links:
                links_item = links_item_data.to_dict()

                links.append(links_item)

        minimum_version = self.minimum_version

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "latestVersion": latest_version,
                "name": name,
                "supported": supported,
            }
        )
        if details_uri is not UNSET:
            field_dict["detailsUri"] = details_uri
        if links is not UNSET:
            field_dict["links"] = links
        if minimum_version is not UNSET:
            field_dict["minimumVersion"] = minimum_version

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "ClientApplication":
        d = src_dict.copy()
        latest_version = d.pop("latestVersion")

        name = d.pop("name")

        supported = d.pop("supported")

        details_uri = d.pop("detailsUri", UNSET)

        links = []
        _links = d.pop("links", UNSET)
        for links_item_data in _links or []:
            links_item = ClientApplicationsDownloadLink.from_dict(links_item_data)

            links.append(links_item)

        minimum_version = d.pop("minimumVersion", UNSET)

        client_application = ClientApplication(
            latest_version=latest_version,
            name=name,
            supported=supported,
            details_uri=details_uri,
            links=links,
            minimum_version=minimum_version,
        )

        client_application.additional_properties = d
        return client_application

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
