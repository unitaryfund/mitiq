from typing import Any, Callable, Dict, List, Optional, Union

import attr

from ..models.validation_error import ValidationError
from ..types import UNSET, Unset
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class Error:
    """  """

    code: str
    message: str
    request_id: str
    validation_errors: Union[Unset, List[ValidationError]] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        code = self.code
        message = self.message
        request_id = self.request_id
        validation_errors: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.validation_errors, Unset):
            validation_errors = []
            for validation_errors_item_data in self.validation_errors:
                validation_errors_item = validation_errors_item_data.to_dict()

                validation_errors.append(validation_errors_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "code": code,
                "message": message,
                "requestId": request_id,
            }
        )
        if validation_errors is not UNSET:
            field_dict["validationErrors"] = validation_errors

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "Error":
        d = src_dict.copy()
        code = d.pop("code")

        message = d.pop("message")

        request_id = d.pop("requestId")

        validation_errors = []
        _validation_errors = d.pop("validationErrors", UNSET)
        for validation_errors_item_data in _validation_errors or []:
            validation_errors_item = ValidationError.from_dict(validation_errors_item_data)

            validation_errors.append(validation_errors_item)

        error = Error(
            code=code,
            message=message,
            request_id=request_id,
            validation_errors=validation_errors,
        )

        error.additional_properties = d
        return error

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
