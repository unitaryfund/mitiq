from typing import Any, Callable, Dict, List, Optional, Union

import attr

from ..types import UNSET, Unset
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class AddGroupUserRequest:
    """ Must provide either `userId` or `userEmail` and `groupId` or `groupName`. """

    group_id: Union[Unset, str] = UNSET
    group_name: Union[Unset, str] = UNSET
    user_email: Union[Unset, str] = UNSET
    user_id: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        group_id = self.group_id
        group_name = self.group_name
        user_email = self.user_email
        user_id = self.user_id

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if group_id is not UNSET:
            field_dict["groupId"] = group_id
        if group_name is not UNSET:
            field_dict["groupName"] = group_name
        if user_email is not UNSET:
            field_dict["userEmail"] = user_email
        if user_id is not UNSET:
            field_dict["userId"] = user_id

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "AddGroupUserRequest":
        d = src_dict.copy()
        group_id = d.pop("groupId", UNSET)

        group_name = d.pop("groupName", UNSET)

        user_email = d.pop("userEmail", UNSET)

        user_id = d.pop("userId", UNSET)

        add_group_user_request = AddGroupUserRequest(
            group_id=group_id,
            group_name=group_name,
            user_email=user_email,
            user_id=user_id,
        )

        add_group_user_request.additional_properties = d
        return add_group_user_request

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
