from typing import Any, Callable, Dict, List, Optional, Union

import attr

from ..models.engagement_credentials import EngagementCredentials
from ..types import UNSET, Unset
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class EngagementWithCredentials:
    """ An engagement is the authorization of a user to execute work on a Quantum Processor Endpoint. """

    address: str
    credentials: EngagementCredentials
    endpoint_id: str
    expires_at: str
    quantum_processor_id: str
    user_id: str
    minimum_priority: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        address = self.address
        credentials = self.credentials.to_dict()

        endpoint_id = self.endpoint_id
        expires_at = self.expires_at
        quantum_processor_id = self.quantum_processor_id
        user_id = self.user_id
        minimum_priority = self.minimum_priority

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "address": address,
                "credentials": credentials,
                "endpointId": endpoint_id,
                "expiresAt": expires_at,
                "quantumProcessorId": quantum_processor_id,
                "userId": user_id,
            }
        )
        if minimum_priority is not UNSET:
            field_dict["minimumPriority"] = minimum_priority

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "EngagementWithCredentials":
        d = src_dict.copy()
        address = d.pop("address")

        credentials = EngagementCredentials.from_dict(d.pop("credentials"))

        endpoint_id = d.pop("endpointId")

        expires_at = d.pop("expiresAt")

        quantum_processor_id = d.pop("quantumProcessorId")

        user_id = d.pop("userId")

        minimum_priority = d.pop("minimumPriority", UNSET)

        engagement_with_credentials = EngagementWithCredentials(
            address=address,
            credentials=credentials,
            endpoint_id=endpoint_id,
            expires_at=expires_at,
            quantum_processor_id=quantum_processor_id,
            user_id=user_id,
            minimum_priority=minimum_priority,
        )

        engagement_with_credentials.additional_properties = d
        return engagement_with_credentials

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
