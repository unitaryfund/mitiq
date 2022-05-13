from typing import Any, Callable, Dict, List, Optional

import attr

from ..models.user_credentials_password import UserCredentialsPassword
from ..types import UNSET
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class UserCredentials:
    """  """

    password: UserCredentialsPassword
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        password = self.password.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "password": password,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "UserCredentials":
        d = src_dict.copy()
        password = UserCredentialsPassword.from_dict(d.pop("password"))

        user_credentials = UserCredentials(
            password=password,
        )

        user_credentials.additional_properties = d
        return user_credentials

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
