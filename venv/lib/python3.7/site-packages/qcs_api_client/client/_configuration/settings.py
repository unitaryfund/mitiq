from typing import Dict
from pydantic import BaseModel
from pydantic.fields import Field
from pydantic.networks import HttpUrl
from qcs_api_client.client._configuration.environment import EnvironmentModel
from qcs_api_client.client._configuration.file import QCSClientConfigurationFile


class QCSAuthServer(BaseModel):
    client_id: str
    issuer: str

    def authorize_url(self):
        return f"{self.issuer}/v1/authorize"

    def token_url(self):
        return f"{self.issuer}/v1/token"

    @staticmethod
    def scopes():
        return ["offline_access"]

    class Config:
        env_prefix = "QCS_SETTINGS_AUTH_SERVER_"


_DEFAULT_AUTH_SERVER = QCSAuthServer(
    client_id="0oa3ykoirzDKpkfzk357",
    issuer="https://auth.qcs.rigetti.com/oauth2/aus8jcovzG0gW2TUG355",
)


class QCSClientConfigurationSettingsApplicationsCLI(EnvironmentModel):
    verbosity: str = ""

    class Config:
        env_prefix = "QCS_SETTINGS_APPLICATIONS_CLI_"


class QCSClientConfigurationSettingsApplicationsPyquil(EnvironmentModel):
    qvm_url: str = "http://127.0.0.1:5000"
    quilc_url: str = "tcp://127.0.0.1:5555"

    class Config:
        env_prefix = "QCS_SETTINGS_APPLICATIONS_PYQUIL_"


class QCSClientConfigurationSettingsApplications(BaseModel):
    """Section of a profile specifying per-application settings."""

    cli: QCSClientConfigurationSettingsApplicationsCLI = Field(
        default_factory=QCSClientConfigurationSettingsApplicationsCLI
    )

    pyquil: QCSClientConfigurationSettingsApplicationsPyquil = Field(
        default_factory=QCSClientConfigurationSettingsApplicationsPyquil
    )


class QCSClientConfigurationSettingsProfile(EnvironmentModel):
    api_url: HttpUrl = "https://api.qcs.rigetti.com"
    """URL of the QCS API to use for all API calls"""

    auth_server_name: str = "default"

    applications: QCSClientConfigurationSettingsApplications = Field(
        default_factory=QCSClientConfigurationSettingsApplications
    )

    credentials_name: str = "default"

    class Config:
        env_prefix = "QCS_SETTINGS_"


class QCSClientConfigurationSettings(QCSClientConfigurationFile):
    default_profile_name: str = "default"
    """Which profile to select settings from when none is specified."""

    profiles: Dict[str, QCSClientConfigurationSettingsProfile] = Field(
        default_factory=lambda: dict(default=QCSClientConfigurationSettingsProfile())
    )
    """All available configuration profiles, keyed by profile name."""

    auth_servers: Dict[str, QCSAuthServer] = Field(default_factory=lambda: {"default": _DEFAULT_AUTH_SERVER})
