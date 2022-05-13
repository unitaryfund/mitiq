from typing import Any, Callable, Dict, List, Optional

import attr

from ..models.parameter_spec import ParameterSpec
from ..types import UNSET
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class TranslateNativeQuilToEncryptedBinaryResponseMemoryDescriptors:
    """  """

    additional_properties: Dict[str, ParameterSpec] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:

        field_dict: Dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        field_dict.update({})

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "TranslateNativeQuilToEncryptedBinaryResponseMemoryDescriptors":
        d = src_dict.copy()
        translate_native_quil_to_encrypted_binary_response_memory_descriptors = (
            TranslateNativeQuilToEncryptedBinaryResponseMemoryDescriptors()
        )

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = ParameterSpec.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        translate_native_quil_to_encrypted_binary_response_memory_descriptors.additional_properties = (
            additional_properties
        )
        return translate_native_quil_to_encrypted_binary_response_memory_descriptors

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> ParameterSpec:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: ParameterSpec) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
