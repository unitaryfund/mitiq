from http import HTTPStatus
from qcs_api_client.util.errors import QCSHTTPStatusError


DEFAULT_RETRY_STATUS_CODES = {
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}


def _is_exception_retryable(exception):
    if isinstance(exception, QCSHTTPStatusError):
        return exception.response.status_code in DEFAULT_RETRY_STATUS_CODES
    return False


DEFAULT_RETRY_ARGUMENTS = {
    "stop_max_attempt_number": 3,
    "wait_exponential_multiplier": 200,
    "wait_exponential_max": 1000,
    "retry_on_exception": _is_exception_retryable,
}
