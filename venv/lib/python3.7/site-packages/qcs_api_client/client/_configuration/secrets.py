import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Union

import toml
from jwt import decode
from pydantic import BaseModel, ValidationError
from pydantic.fields import Field

from qcs_api_client.client._configuration.file import QCSClientConfigurationFile

_TOKEN_REFRESH_PREEMPT_INTERVAL = timedelta(seconds=15)
_secrets_write_lock = threading.RLock()


class TokenPayload(BaseModel):
    """
    TokenPayload represents a response from the OAuth2 POST /token endpoint.
    """

    refresh_token: Optional[str]
    access_token: Optional[str]
    scope: Optional[str]
    expires_in: Optional[int]
    id_token: Optional[str]
    token_type: Optional[str]

    def get_access_token_claims(self, key: Union[None, bytes, str] = None):
        """
        Return the claims within the encoded access token.

        If a JWK is provided as ``key``, verify the claims as well. If no key is provided, be aware
        that the returned claims might be forged or invalid.
        """
        if key is None:
            return decode(self.access_token, verify=False)

        return decode(self.access_token, key=key)

    @property
    def access_token_expires_at(self) -> Optional[datetime]:
        """
        Return the datetime that the token expires (if any).
        """
        claims = self.get_access_token_claims()
        iat = claims.get("iat")
        if iat is None:
            return None
        return datetime.utcfromtimestamp(int(iat))

    def should_refresh(self) -> bool:
        """
        Return True if the token is past or nearing expiration and should be refreshed from the
        auth server.
        """
        iat = self.access_token_expires_at
        if iat is None:
            return False
        return iat - _TOKEN_REFRESH_PREEMPT_INTERVAL < datetime.now().astimezone(timezone.utc)


class QCSClientConfigurationSecretsCredentials(BaseModel):
    token_payload: Optional[TokenPayload] = None

    @property
    def access_token(self) -> Optional[str]:
        if self.token_payload is not None:
            return self.token_payload.access_token

    @property
    def refresh_token(self) -> Optional[str]:
        if self.token_payload is not None:
            return self.token_payload.refresh_token


class QCSClientConfigurationSecrets(QCSClientConfigurationFile):
    credentials: Dict[str, QCSClientConfigurationSecretsCredentials] = Field(
        default_factory=lambda: dict(default=QCSClientConfigurationSecretsCredentials())
    )

    def update_token(self, *, credentials_name: str, token: TokenPayload) -> None:
        """
        Update the value of a token payload in memory and (if appropriate) on disk.
        """
        with _secrets_write_lock:
            if credentials_name not in self.credentials:
                self.credentials[credentials_name] = QCSClientConfigurationSecretsCredentials()
            self.credentials[credentials_name].token_payload = token

            self._write_token_to_file(credentials_name=credentials_name, token=token)

    def _write_token_to_file(self, *, credentials_name: str, token: TokenPayload) -> None:
        """
        Update the value of a token payload within the file, if a file path is provided.
        """
        if self.file_path is None:
            return

        with open(self.file_path, "r") as f:
            current_data = toml.load(f)

        try:
            # Parse the file to validate its structure
            QCSClientConfigurationSecrets(**current_data)

        except ValidationError:
            raise RuntimeError(
                f"Unable to write credentials back to {self.file_path}; file is not in valid QCS secrets format"
            )

        if "credentials" not in current_data:
            current_data["credentials"] = {}

        if credentials_name not in current_data["credentials"]:
            current_data["credentials"][credentials_name] = {}

        current_data["credentials"][credentials_name]["token_payload"] = token.dict()

        with open(self.file_path, "w") as f:
            toml.dump(current_data, f)
