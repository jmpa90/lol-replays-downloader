"""Minimal offline unit test for download_replay.extract_match_id().

Plain asserts, no test framework / network required:

    python tests/test_extract_match_id.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from download_replay import extract_match_id  # noqa: E402


def test_current_url_shape():
    # Current Riot replay URL shape: /{matchId}/0.replay
    url = (
        "https://gamereplay.dyn.riotcdn.net/replay/kr/1/369/"
        "KR_8016115724/0.replay?token=abc123signed"
    )
    assert extract_match_id(url) == "KR_8016115724"


def test_older_url_shape():
    # Older shape: /{matchId}.replay (no /0.replay segment)
    url = "https://gamereplay.dyn.riotcdn.net/replay/kr/1/369/KR_8015940054.replay"
    assert extract_match_id(url) == "KR_8015940054"


def test_no_match_returns_none():
    # Regression case: the qsxmiocmio#KR1 (369) URL shape that previously
    # crashed with 'NoneType' object has no attribute 'group'.
    url = "https://gamereplay.dyn.riotcdn.net/replay/kr/1/369/weird-shape"
    assert extract_match_id(url) is None


def test_no_match_falls_back_to_metadata():
    url = "https://gamereplay.dyn.riotcdn.net/replay/kr/1/369/weird-shape"
    assert extract_match_id(url, metadata={"matchId": "kr_369"}) == "KR_369"


def test_empty_url_no_metadata_returns_none():
    assert extract_match_id("") is None
    assert extract_match_id(None) is None


if __name__ == "__main__":
    test_current_url_shape()
    test_older_url_shape()
    test_no_match_returns_none()
    test_no_match_falls_back_to_metadata()
    test_empty_url_no_metadata_returns_none()
    print("All extract_match_id tests passed.")
