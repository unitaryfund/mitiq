import datetime
from typing import Any, Callable, Dict, List, Optional, Union, cast

import attr
from dateutil.parser import isoparse
from rfc3339 import rfc3339

from ..types import UNSET, Unset
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class Reservation:
    """  """

    created_time: datetime.datetime
    end_time: datetime.datetime
    id: int
    price: int
    quantum_processor_id: str
    start_time: datetime.datetime
    user_id: str
    cancelled: Union[Unset, bool] = UNSET
    notes: Union[Unset, str] = UNSET
    updated_time: Union[Unset, datetime.datetime] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        assert self.created_time.tzinfo is not None, "Datetime must have timezone information"
        created_time = rfc3339(self.created_time)

        assert self.end_time.tzinfo is not None, "Datetime must have timezone information"
        end_time = rfc3339(self.end_time)

        id = self.id
        price = self.price
        quantum_processor_id = self.quantum_processor_id
        assert self.start_time.tzinfo is not None, "Datetime must have timezone information"
        start_time = rfc3339(self.start_time)

        user_id = self.user_id
        cancelled = self.cancelled
        notes = self.notes
        updated_time: Union[Unset, str] = UNSET
        if not isinstance(self.updated_time, Unset):
            assert self.updated_time.tzinfo is not None, "Datetime must have timezone information"
            updated_time = rfc3339(self.updated_time)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "createdTime": created_time,
                "endTime": end_time,
                "id": id,
                "price": price,
                "quantumProcessorId": quantum_processor_id,
                "startTime": start_time,
                "userId": user_id,
            }
        )
        if cancelled is not UNSET:
            field_dict["cancelled"] = cancelled
        if notes is not UNSET:
            field_dict["notes"] = notes
        if updated_time is not UNSET:
            field_dict["updatedTime"] = updated_time

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "Reservation":
        d = src_dict.copy()
        created_time = isoparse(d.pop("createdTime"))

        end_time = isoparse(d.pop("endTime"))

        id = d.pop("id")

        price = d.pop("price")

        quantum_processor_id = d.pop("quantumProcessorId")

        start_time = isoparse(d.pop("startTime"))

        user_id = d.pop("userId")

        cancelled = d.pop("cancelled", UNSET)

        notes = d.pop("notes", UNSET)

        updated_time = None
        if d.pop("updatedTime", UNSET) is not None:
            updated_time = isoparse(cast(str, d.pop("updatedTime", UNSET)))

        reservation = Reservation(
            created_time=created_time,
            end_time=end_time,
            id=id,
            price=price,
            quantum_processor_id=quantum_processor_id,
            start_time=start_time,
            user_id=user_id,
            cancelled=cancelled,
            notes=notes,
            updated_time=updated_time,
        )

        reservation.additional_properties = d
        return reservation

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
