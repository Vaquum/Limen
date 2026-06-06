from io import BytesIO

import polars as pl
import pytest

from limen.data import historical_data
from limen.data.historical_data import _read_any_file
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


def test_read_any_file_routes_remote_parquet_through_retrying_reader() -> None:
    frame = pl.DataFrame({'a': [1, 2, 3]})
    buffer = BytesIO()
    frame.write_parquet(buffer)
    parquet_bytes = buffer.getvalue()
    seen: list[str] = []

    def fake_read_remote_bytes(url: str) -> bytes:
        seen.append(url)
        return parquet_bytes

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(historical_data, '_read_remote_bytes', fake_read_remote_bytes)
        result = _read_any_file('https://example.com/data.parquet')

    assert seen == ['https://example.com/data.parquet']
    assert result.equals(frame)
