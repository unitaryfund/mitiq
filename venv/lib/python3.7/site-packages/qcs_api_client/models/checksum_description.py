from typing import Any, Callable, Dict, List, Optional

import attr

from ..models.checksum_description_type import ChecksumDescriptionType
from ..types import UNSET
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class ChecksumDescription:
    """  """

    header_name: str
    type: ChecksumDescriptionType
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        header_name = self.header_name
        type = self.type.value

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "headerName": header_name,
                "type": type,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "ChecksumDescription":
        d = src_dict.copy()
        header_name = d.pop("headerName")

        type = ChecksumDescriptionType(d.pop("type"))

        checksum_description = ChecksumDescription(
            header_name=header_name,
            type=type,
        )

        checksum_description.additional_properties = d
        return checksum_description

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
