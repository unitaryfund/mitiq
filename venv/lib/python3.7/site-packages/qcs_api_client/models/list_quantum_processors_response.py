from typing import Any, Callable, Dict, List, Optional, Union

import attr

from ..models.quantum_processor import QuantumProcessor
from ..types import UNSET, Unset
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class ListQuantumProcessorsResponse:
    """  """

    quantum_processors: List[QuantumProcessor]
    next_page_token: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        quantum_processors = []
        for quantum_processors_item_data in self.quantum_processors:
            quantum_processors_item = quantum_processors_item_data.to_dict()

            quantum_processors.append(quantum_processors_item)

        next_page_token = self.next_page_token

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "quantumProcessors": quantum_processors,
            }
        )
        if next_page_token is not UNSET:
            field_dict["nextPageToken"] = next_page_token

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "ListQuantumProcessorsResponse":
        d = src_dict.copy()
        quantum_processors = []
        _quantum_processors = d.pop("quantumProcessors")
        for quantum_processors_item_data in _quantum_processors:
            quantum_processors_item = QuantumProcessor.from_dict(quantum_processors_item_data)

            quantum_processors.append(quantum_processors_item)

        next_page_token = d.pop("nextPageToken", UNSET)

        list_quantum_processors_response = ListQuantumProcessorsResponse(
            quantum_processors=quantum_processors,
            next_page_token=next_page_token,
        )

        list_quantum_processors_response.additional_properties = d
        return list_quantum_processors_response

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
