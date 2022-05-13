from typing import Any, Callable, Dict, List, Optional

import attr

from ..models.client_application import ClientApplication
from ..types import UNSET
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class ListClientApplicationsResponse:
    """  """

    client_applications: List[ClientApplication]
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        client_applications = []
        for client_applications_item_data in self.client_applications:
            client_applications_item = client_applications_item_data.to_dict()

            client_applications.append(client_applications_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "clientApplications": client_applications,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "ListClientApplicationsResponse":
        d = src_dict.copy()
        client_applications = []
        _client_applications = d.pop("clientApplications")
        for client_applications_item_data in _client_applications:
            client_applications_item = ClientApplication.from_dict(client_applications_item_data)

            client_applications.append(client_applications_item)

        list_client_applications_response = ListClientApplicationsResponse(
            client_applications=client_applications,
        )

        list_client_applications_response.additional_properties = d
        return list_client_applications_response

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
