from typing import Any, Callable, Dict, List, Optional

import attr

from ..types import UNSET
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class Endpoint:
    """ An Endpoint is the entry point for remote access to a QuantumProcessor. """

    address: str
    healthy: bool
    id: str
    mock: bool
    quantum_processor_id: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        address = self.address
        healthy = self.healthy
        id = self.id
        mock = self.mock
        quantum_processor_id = self.quantum_processor_id

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "address": address,
                "healthy": healthy,
                "id": id,
                "mock": mock,
                "quantumProcessorId": quantum_processor_id,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "Endpoint":
        d = src_dict.copy()
        address = d.pop("address")

        healthy = d.pop("healthy")

        id = d.pop("id")

        mock = d.pop("mock")

        quantum_processor_id = d.pop("quantumProcessorId")

        endpoint = Endpoint(
            address=address,
            healthy=healthy,
            id=id,
            mock=mock,
            quantum_processor_id=quantum_processor_id,
        )

        endpoint.additional_properties = d
        return endpoint

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
