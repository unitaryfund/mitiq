from enum import Enum


class Family(str, Enum):
    ASPEN = "Aspen"

    def __str__(self) -> str:
        return str(self.value)
