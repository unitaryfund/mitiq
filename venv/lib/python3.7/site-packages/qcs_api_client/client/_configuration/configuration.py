import os
from pathlib import Path
from typing import Optional

from pydantic.main import BaseModel

from qcs_api_client.client._configuration.error import QCSClientConfigurationError
from qcs_api_client.client._configuration.secrets import (
    QCSClientConfigurationSecrets,
    QCSClientConfigurationSecretsCredentials,
)
from qcs_api_client.client._configuration.settings import (
    QCSClientConfigurationSettings,
    QCSClientConfigurationSettingsProfile,
)


QCS_BASE_PATH = Path("~/.qcs").expanduser()

DEFAULT_SECRETS_FILE_PATH = QCS_BASE_PATH / "secrets.toml"
DEFAULT_SETTINGS_FILE_PATH = QCS_BASE_PATH / "settings.toml"


class QCSClientConfiguration(BaseModel):
    profile_name: str
    secrets: QCSClientConfigurationSecrets
    settings: QCSClientConfigurationSettings

    @property
    def auth_server(self):
        server = self.settings.auth_servers.get(self.profile.auth_server_name)
        if server is None:
            raise QCSClientConfigurationError(f"no authorization server configured for {self.profile.auth_server_name}")

        return server

    @property
    def credentials(self) -> QCSClientConfigurationSecretsCredentials:
        # return self.secrets.credentials[self.profile.credentials_name]
        credentials = self.secrets.credentials.get(self.profile.credentials_name)
        if credentials is None:
            raise QCSClientConfigurationError(f"no credentials available named '{self.profile.credentials_name}'")
        return credentials

    @property
    def profile(self) -> QCSClientConfigurationSettingsProfile:
        profile = self.settings.profiles.get(self.profile_name)
        if profile is None:
            raise QCSClientConfigurationError(f"no profile available named '{self.profile_name}'")
        return profile

    @classmethod
    def load(
        cls,
        profile_name: Optional[str] = None,
        settings_file_path: Optional[os.PathLike] = None,
        secrets_file_path: Optional[os.PathLike] = None,
    ) -> "QCSClientConfiguration":
        secrets_file_path = secrets_file_path or os.getenv("QCS_SECRETS_FILE_PATH", DEFAULT_SECRETS_FILE_PATH)

        secrets = QCSClientConfigurationSecrets(file_path=secrets_file_path)

        settings_file_path = settings_file_path or os.getenv("QCS_SETTINGS_FILE_PATH", DEFAULT_SETTINGS_FILE_PATH)

        settings = QCSClientConfigurationSettings(file_path=settings_file_path)

        profile_name = profile_name or os.getenv("QCS_PROFILE_NAME", settings.default_profile_name)

        return cls(profile_name=profile_name, secrets=secrets, settings=settings)
