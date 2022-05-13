from enum import Enum


class ChecksumDescriptionType(str, Enum):
    M_D5 = "md5"

    def __str__(self) -> str:
        return str(self.value)
