from pathlib import Path
from typing import Any, Dict, Optional

import toml
from pydantic.types import FilePath
from pydantic.utils import deep_update

from qcs_api_client.client._configuration.environment import _EnvironmentBaseModel


class QCSClientConfigurationFile(_EnvironmentBaseModel):
    """
    A configuration value store which loads field values in this order of precedence on
    initialization:
    1. Runtime keyword arguments
    2. Environment variables
    3. Configuration file path (if provided)
    4. Field default values
    """

    file_path: Optional[Path] = None
    """
    The file path which maps to this configuration, and to which this configuration will be written
    on save.
    """

    def __init__(self, **kwargs: Any) -> None:
        values = deep_update(
            self._read_file(kwargs.get("file_path")),
            self._read_env(),
            kwargs,
        )

        super().__init__(**values)

    def _read_file(self, file_path: Optional[Path]) -> Dict[str, Any]:
        """
        Read a dict from a file path if it exists.
        """
        if file_path is None:
            return {}

        path = Path(file_path)
        if not path.exists():
            return {}

        with open(path, "r") as f:
            return toml.load(f)

    def dict(self, *args, **kwargs) -> dict:
        if "exclude" in kwargs:
            kwargs["exclude"].add("file_path")
        else:
            kwargs["exclude"] = {"file_path"}

        return super().dict(*args, **kwargs)

    def json(self, *args, **kwargs) -> str:
        if "exclude" in kwargs:
            kwargs["exclude"].add("file_path")
        else:
            kwargs["exclude"] = {"file_path"}

        return super().json(*args, **kwargs)

    @classmethod
    def parse_file(cls, path: FilePath) -> "QCSClientConfigurationFile":
        with open(path, "r") as f:
            return cls(file_path=path, **toml.load(f))
