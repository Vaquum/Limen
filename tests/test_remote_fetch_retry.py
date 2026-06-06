from limen.data.historical_data import _REMOTE_BACKOFF_FACTOR
from limen.data.historical_data import _REMOTE_MAX_RETRIES
from limen.data.historical_data import _REMOTE_RETRY_STATUSES
from limen.data.historical_data import _remote_session


def test_remote_session_is_configured_to_retry_transient_errors() -> None:
    with _remote_session() as session:
        for scheme in ('https://', 'http://'):
            retry = session.get_adapter(f'{scheme}example').max_retries
            assert retry.total == _REMOTE_MAX_RETRIES
            assert retry.backoff_factor == _REMOTE_BACKOFF_FACTOR
            assert set(_REMOTE_RETRY_STATUSES) <= set(retry.status_forcelist)
            assert 429 in retry.status_forcelist
            assert retry.respect_retry_after_header is True
            assert 'GET' in retry.allowed_methods
