from typing import Any, Callable, Dict, List, Optional

import attr

from ..types import UNSET
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class AuthResetPasswordRequest:
    """  """

    new_password: str
    old_password: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        new_password = self.new_password
        old_password = self.old_password

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "newPassword": new_password,
                "oldPassword": old_password,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "AuthResetPasswordRequest":
        d = src_dict.copy()
        new_password = d.pop("newPassword")

        old_password = d.pop("oldPassword")

        auth_reset_password_request = AuthResetPasswordRequest(
            new_password=new_password,
            old_password=old_password,
        )

        auth_reset_password_request.additional_properties = d
        return auth_reset_password_request

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
