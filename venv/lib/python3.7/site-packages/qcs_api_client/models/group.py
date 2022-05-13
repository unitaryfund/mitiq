import datetime
from typing import Any, Callable, Dict, List, Optional

import attr
from dateutil.parser import isoparse
from rfc3339 import rfc3339

from ..types import UNSET
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class Group:
    """  """

    created_time: datetime.datetime
    description: str
    id: str
    last_membership_updated_time: datetime.datetime
    name: str
    updated_time: datetime.datetime
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        assert self.created_time.tzinfo is not None, "Datetime must have timezone information"
        created_time = rfc3339(self.created_time)

        description = self.description
        id = self.id
        assert self.last_membership_updated_time.tzinfo is not None, "Datetime must have timezone information"
        last_membership_updated_time = rfc3339(self.last_membership_updated_time)

        name = self.name
        assert self.updated_time.tzinfo is not None, "Datetime must have timezone information"
        updated_time = rfc3339(self.updated_time)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "createdTime": created_time,
                "description": description,
                "id": id,
                "lastMembershipUpdatedTime": last_membership_updated_time,
                "name": name,
                "updatedTime": updated_time,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "Group":
        d = src_dict.copy()
        created_time = isoparse(d.pop("createdTime"))

        description = d.pop("description")

        id = d.pop("id")

        last_membership_updated_time = isoparse(d.pop("lastMembershipUpdatedTime"))

        name = d.pop("name")

        updated_time = isoparse(d.pop("updatedTime"))

        group = Group(
            created_time=created_time,
            description=description,
            id=id,
            last_membership_updated_time=last_membership_updated_time,
            name=name,
            updated_time=updated_time,
        )

        group.additional_properties = d
        return group

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
