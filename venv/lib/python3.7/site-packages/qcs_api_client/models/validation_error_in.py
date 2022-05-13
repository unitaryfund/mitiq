from enum import Enum


class ValidationErrorIn(str, Enum):
    HEADER = "header"
    QUERY = "query"
    PATH = "path"
    BODY = "body"

    def __str__(self) -> str:
        return str(self.value)
