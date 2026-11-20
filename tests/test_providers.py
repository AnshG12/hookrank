import pytest

from hookrank.providers import ProviderError, extract_json


def test_extract_json_accepts_fenced_object_and_rejects_array():
    assert extract_json('```json\n{"ok":true}\n```') == {"ok": True}
    with pytest.raises(ProviderError):
        extract_json('[{"ok":true}]')
