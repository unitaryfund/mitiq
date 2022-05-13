from typing import Any, Callable, Dict, List, Optional, Union

import attr

from ..types import UNSET, Unset
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class TranslateNativeQuilToEncryptedBinaryRequest:
    """  """

    num_shots: int
    quil: str
    settings_timestamp: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        num_shots = self.num_shots
        quil = self.quil
        settings_timestamp = self.settings_timestamp

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "numShots": num_shots,
                "quil": quil,
            }
        )
        if settings_timestamp is not UNSET:
            field_dict["settingsTimestamp"] = settings_timestamp

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "TranslateNativeQuilToEncryptedBinaryRequest":
        d = src_dict.copy()
        num_shots = d.pop("numShots")

        quil = d.pop("quil")

        settings_timestamp = d.pop("settingsTimestamp", UNSET)

        translate_native_quil_to_encrypted_binary_request = TranslateNativeQuilToEncryptedBinaryRequest(
            num_shots=num_shots,
            quil=quil,
            settings_timestamp=settings_timestamp,
        )

        translate_native_quil_to_encrypted_binary_request.additional_properties = d
        return translate_native_quil_to_encrypted_binary_request

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
