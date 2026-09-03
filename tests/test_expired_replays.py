"""Minimal offline unit tests for download_replay's expired-replay cache and
age-filter helpers. Plain asserts, no test framework / network required:

    python tests/test_expired_replays.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from download_replay import (  # noqa: E402
    _max_age_start_time,
    _should_skip,
    load_expired_replays,
    save_expired_replays,
)


def test_max_age_start_time_is_n_days_before_now():
    now = 1_700_000_000  # fixed reference instant (epoch seconds)
    cutoff = _max_age_start_time(14, now=now)
    assert cutoff == now - 14 * 86400


def test_max_age_start_time_defaults_to_wall_clock():
    # Just check it returns a plausible epoch-seconds int without crashing.
    cutoff = _max_age_start_time(14)
    assert isinstance(cutoff, int)
    assert cutoff > 0


def test_should_skip_known_expired():
    assert _should_skip("KR_123", expired_ids={"KR_123"}, recent_ids=None) is True


def test_should_skip_outside_recent_window():
    assert _should_skip("KR_123", expired_ids=set(), recent_ids={"KR_999"}) is True


def test_should_skip_inside_recent_window():
    assert _should_skip("KR_123", expired_ids=set(), recent_ids={"KR_123"}) is False


def test_should_skip_no_recent_window_known_not_expired():
    # recent_ids=None means "age unknown, don't filter by age"
    assert _should_skip("KR_123", expired_ids=set(), recent_ids=None) is False


def test_load_expired_replays_missing_file_returns_empty_set():
    missing_path = os.path.join(tempfile.gettempdir(), "does_not_exist_expired.json")
    if os.path.exists(missing_path):
        os.remove(missing_path)
    assert load_expired_replays(path=missing_path) == set()


def test_load_expired_replays_invalid_json_returns_empty_set():
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("not valid json{")
        assert load_expired_replays(path=path) == set()
    finally:
        os.remove(path)


def test_save_and_load_roundtrip():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        ids = {"KR_1", "EUW1_2", "KR_1"}  # duplicate collapses via set
        save_expired_replays(ids, path=path)

        with open(path, "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk == sorted(ids)  # saved sorted, for stable diffs

        loaded = load_expired_replays(path=path)
        assert loaded == ids
    finally:
        os.remove(path)


if __name__ == "__main__":
    test_max_age_start_time_is_n_days_before_now()
    test_max_age_start_time_defaults_to_wall_clock()
    test_should_skip_known_expired()
    test_should_skip_outside_recent_window()
    test_should_skip_inside_recent_window()
    test_should_skip_no_recent_window_known_not_expired()
    test_load_expired_replays_missing_file_returns_empty_set()
    test_load_expired_replays_invalid_json_returns_empty_set()
    test_save_and_load_roundtrip()
    print("All expired-replays/age-filter tests passed.")
