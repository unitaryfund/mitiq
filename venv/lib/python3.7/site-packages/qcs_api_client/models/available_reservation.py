import datetime
from typing import Any, Callable, Dict, List, Optional

import attr
from dateutil.parser import isoparse
from rfc3339 import rfc3339

from ..types import UNSET
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class AvailableReservation:
    """  """

    duration: str
    end_time: datetime.datetime
    price: int
    quantum_processor_id: str
    start_time: datetime.datetime
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        duration = self.duration
        assert self.end_time.tzinfo is not None, "Datetime must have timezone information"
        end_time = rfc3339(self.end_time)

        price = self.price
        quantum_processor_id = self.quantum_processor_id
        assert self.start_time.tzinfo is not None, "Datetime must have timezone information"
        start_time = rfc3339(self.start_time)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "duration": duration,
                "endTime": end_time,
                "price": price,
                "quantumProcessorId": quantum_processor_id,
                "startTime": start_time,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "AvailableReservation":
        d = src_dict.copy()
        duration = d.pop("duration")

        end_time = isoparse(d.pop("endTime"))

        price = d.pop("price")

        quantum_processor_id = d.pop("quantumProcessorId")

        start_time = isoparse(d.pop("startTime"))

        available_reservation = AvailableReservation(
            duration=duration,
            end_time=end_time,
            price=price,
            quantum_processor_id=quantum_processor_id,
            start_time=start_time,
        )

        available_reservation.additional_properties = d
        return available_reservation

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
