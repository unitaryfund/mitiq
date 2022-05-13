from typing import Any, Callable, Dict, List, Optional, Union

import attr

from ..types import UNSET, Unset
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class CreateEngagementRequest:
    """  """

    endpoint_id: Union[Unset, str] = UNSET
    quantum_processor_id: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        endpoint_id = self.endpoint_id
        quantum_processor_id = self.quantum_processor_id

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if endpoint_id is not UNSET:
            field_dict["endpointId"] = endpoint_id
        if quantum_processor_id is not UNSET:
            field_dict["quantumProcessorId"] = quantum_processor_id

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "CreateEngagementRequest":
        d = src_dict.copy()
        endpoint_id = d.pop("endpointId", UNSET)

        quantum_processor_id = d.pop("quantumProcessorId", UNSET)

        create_engagement_request = CreateEngagementRequest(
            endpoint_id=endpoint_id,
            quantum_processor_id=quantum_processor_id,
        )

        create_engagement_request.additional_properties = d
        return create_engagement_request

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
