from typing import Any, Callable, Dict, List, Optional

import attr

from ..types import UNSET
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class CheckClientApplicationResponse:
    """  """

    is_latest_version: bool
    is_update_required: bool
    message: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        is_latest_version = self.is_latest_version
        is_update_required = self.is_update_required
        message = self.message

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "isLatestVersion": is_latest_version,
                "isUpdateRequired": is_update_required,
                "message": message,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "CheckClientApplicationResponse":
        d = src_dict.copy()
        is_latest_version = d.pop("isLatestVersion")

        is_update_required = d.pop("isUpdateRequired")

        message = d.pop("message")

        check_client_application_response = CheckClientApplicationResponse(
            is_latest_version=is_latest_version,
            is_update_required=is_update_required,
            message=message,
        )

        check_client_application_response.additional_properties = d
        return check_client_application_response

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
