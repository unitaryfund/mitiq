from typing import Any, Callable, Dict, List, Optional, Union, cast

import attr

from ..models.translate_native_quil_to_encrypted_binary_response_memory_descriptors import (
    TranslateNativeQuilToEncryptedBinaryResponseMemoryDescriptors,
)
from ..types import UNSET, Unset
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class TranslateNativeQuilToEncryptedBinaryResponse:
    """  """

    program: str
    memory_descriptors: Union[TranslateNativeQuilToEncryptedBinaryResponseMemoryDescriptors, Unset] = UNSET
    ro_sources: Union[Unset, List[List[str]]] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        program = self.program
        memory_descriptors: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.memory_descriptors, Unset):
            memory_descriptors = self.memory_descriptors.to_dict()

        ro_sources: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.ro_sources, Unset):
            ro_sources = []
            for ro_sources_item_data in self.ro_sources:
                ro_sources_item = ro_sources_item_data

                ro_sources.append(ro_sources_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "program": program,
            }
        )
        if memory_descriptors is not UNSET:
            field_dict["memoryDescriptors"] = memory_descriptors
        if ro_sources is not UNSET:
            field_dict["roSources"] = ro_sources

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "TranslateNativeQuilToEncryptedBinaryResponse":
        d = src_dict.copy()
        program = d.pop("program")

        memory_descriptors: Union[TranslateNativeQuilToEncryptedBinaryResponseMemoryDescriptors, Unset] = UNSET
        _memory_descriptors = d.pop("memoryDescriptors", UNSET)
        if _memory_descriptors is not None and not isinstance(_memory_descriptors, Unset):
            memory_descriptors = TranslateNativeQuilToEncryptedBinaryResponseMemoryDescriptors.from_dict(
                cast(Dict[str, Any], _memory_descriptors)
            )

        ro_sources = []
        _ro_sources = d.pop("roSources", UNSET)
        for ro_sources_item_data in _ro_sources or []:
            ro_sources_item = cast(List[str], ro_sources_item_data)

            ro_sources.append(ro_sources_item)

        translate_native_quil_to_encrypted_binary_response = TranslateNativeQuilToEncryptedBinaryResponse(
            program=program,
            memory_descriptors=memory_descriptors,
            ro_sources=ro_sources,
        )

        translate_native_quil_to_encrypted_binary_response.additional_properties = d
        return translate_native_quil_to_encrypted_binary_response

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
