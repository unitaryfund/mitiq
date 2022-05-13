from typing import Any, Callable, Dict, List, Optional

import attr

from ..models.reservation import Reservation
from ..types import UNSET
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class ListReservationsResponse:
    """  """

    next_page_token: str
    reservations: List[Reservation]
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        next_page_token = self.next_page_token
        reservations = []
        for reservations_item_data in self.reservations:
            reservations_item = reservations_item_data.to_dict()

            reservations.append(reservations_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "nextPageToken": next_page_token,
                "reservations": reservations,
            }
        )

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "ListReservationsResponse":
        d = src_dict.copy()
        next_page_token = d.pop("nextPageToken")

        reservations = []
        _reservations = d.pop("reservations")
        for reservations_item_data in _reservations:
            reservations_item = Reservation.from_dict(reservations_item_data)

            reservations.append(reservations_item)

        list_reservations_response = ListReservationsResponse(
            next_page_token=next_page_token,
            reservations=reservations,
        )

        list_reservations_response.additional_properties = d
        return list_reservations_response

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
