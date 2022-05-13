from typing import Any, Callable, Dict, List, Optional, Union, cast

import attr

from ..models.validation_error_in import ValidationErrorIn
from ..types import UNSET, Unset
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class ValidationError:
    """  """

    in_: ValidationErrorIn
    message: str
    path: Union[Unset, List[str]] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        in_ = self.in_.value

        message = self.message
        path: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.path, Unset):
            path = self.path

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "in": in_,
                "message": message,
            }
        )
        if path is not UNSET:
            field_dict["path"] = path

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "ValidationError":
        d = src_dict.copy()
        in_ = ValidationErrorIn(d.pop("in"))

        message = d.pop("message")

        path = cast(List[str], d.pop("path", UNSET))

        validation_error = ValidationError(
            in_=in_,
            message=message,
            path=path,
        )

        validation_error.additional_properties = d
        return validation_error

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
