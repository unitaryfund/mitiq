from typing import Any, Callable, Dict, List, Optional, Union

import attr

from ..models.available_reservation import AvailableReservation
from ..types import UNSET, Unset
from ..util.serialization import is_not_none


@attr.s(auto_attribs=True)
class FindAvailableReservationsResponse:
    """  """

    available_reservations: List[AvailableReservation]
    next_page_token: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self, pick_by_predicate: Optional[Callable[[Any], bool]] = is_not_none) -> Dict[str, Any]:
        available_reservations = []
        for available_reservations_item_data in self.available_reservations:
            available_reservations_item = available_reservations_item_data.to_dict()

            available_reservations.append(available_reservations_item)

        next_page_token = self.next_page_token

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "availableReservations": available_reservations,
            }
        )
        if next_page_token is not UNSET:
            field_dict["nextPageToken"] = next_page_token

        field_dict = {k: v for k, v in field_dict.items() if v != UNSET}
        if pick_by_predicate is not None:
            field_dict = {k: v for k, v in field_dict.items() if pick_by_predicate(v)}

        return field_dict

    @staticmethod
    def from_dict(src_dict: Dict[str, Any]) -> "FindAvailableReservationsResponse":
        d = src_dict.copy()
        available_reservations = []
        _available_reservations = d.pop("availableReservations")
        for available_reservations_item_data in _available_reservations:
            available_reservations_item = AvailableReservation.from_dict(available_reservations_item_data)

            available_reservations.append(available_reservations_item)

        next_page_token = d.pop("nextPageToken", UNSET)

        find_available_reservations_response = FindAvailableReservationsResponse(
            available_reservations=available_reservations,
            next_page_token=next_page_token,
        )

        find_available_reservations_response.additional_properties = d
        return find_available_reservations_response

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
