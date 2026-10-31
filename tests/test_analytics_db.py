"""Tests for analytics_db.AnalyticsDatabase.

Each test uses its own throwaway sqlite file (not the shared production
analytics.db) via AnalyticsDatabase(db_path=...).
"""
import os
import sqlite3
from datetime import datetime, timedelta

import pytest

from analytics_db import AnalyticsDatabase


@pytest.fixture
def db(tmp_path):
    return AnalyticsDatabase(db_path=str(tmp_path / "test_analytics.db"))


def _insert_access_log_at(db, timestamp):
    with db._get_connection() as conn:
        conn.execute(
            "INSERT INTO access_logs (timestamp, path, method, status_code, client_host, user_agent) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, "/", "GET", 200, "127.0.0.1", "pytest"),
        )
        conn.commit()


def _insert_analysis_event_at(db, timestamp):
    with db._get_connection() as conn:
        conn.execute(
            "INSERT INTO analysis_events (timestamp, event_type, filename) VALUES (?, ?, ?)",
            (timestamp, "sample_load", "Cu.cif"),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# purge_old_data -- regression test for S-11 (analytics.db had no
# retention/purge at all; IPs and User-Agents accumulated forever)
# ---------------------------------------------------------------------------

def test_purge_old_data_removes_rows_past_retention_and_keeps_recent(db):
    now = datetime.now()
    old = (now - timedelta(days=100)).strftime('%Y-%m-%d %H:%M:%S')
    recent = (now - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')

    _insert_access_log_at(db, old)
    _insert_access_log_at(db, recent)
    _insert_analysis_event_at(db, old)
    _insert_analysis_event_at(db, recent)

    deleted = db.purge_old_data(retention_days=90)
    assert deleted == 2

    with db._get_connection() as conn:
        access_count = conn.execute("SELECT COUNT(*) FROM access_logs").fetchone()[0]
        events_count = conn.execute("SELECT COUNT(*) FROM analysis_events").fetchone()[0]
    assert access_count == 1
    assert events_count == 1


def test_purge_old_data_is_a_noop_when_nothing_is_old(db):
    recent = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
    _insert_access_log_at(db, recent)

    deleted = db.purge_old_data(retention_days=90)
    assert deleted == 0

    with db._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM access_logs").fetchone()[0] == 1


def test_purge_old_data_returns_zero_on_db_error(tmp_path):
    """purge_old_data must degrade gracefully (matches every other method
    in this class), not raise, if the underlying query fails."""
    db = AnalyticsDatabase(db_path=str(tmp_path / "broken.db"))
    with db._get_connection() as conn:
        conn.execute("DROP TABLE access_logs")
        conn.commit()

    assert db.purge_old_data(retention_days=90) == 0


# ---------------------------------------------------------------------------
# UTC consistency -- regression test for C6 (purge_old_data and
# get_daily_access_counts previously used datetime.now(), i.e. the server's
# local time, while timestamps are inserted via SQLite's CURRENT_TIMESTAMP,
# which is always UTC; on a machine ahead of UTC this made both methods
# treat rows as older/newer than they really are). Both now use
# datetime.utcnow(), so fixture timestamps here are built the same way --
# the point is that the comparison is internally consistent in UTC
# regardless of the local timezone offset the test machine happens to have.
# ---------------------------------------------------------------------------

def test_get_daily_access_counts_uses_utc_threshold(db):
    """A row timestamped 1 hour ago in UTC must be counted by
    get_daily_access_counts(days=1), which computes its threshold from
    datetime.utcnow() (not datetime.now())."""
    one_hour_ago_utc = (datetime.utcnow() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    _insert_access_log_at(db, one_hour_ago_utc)

    counts = db.get_daily_access_counts(days=1)
    total = sum(row["count"] for row in counts)
    assert total == 1


# ---------------------------------------------------------------------------
# Test isolation -- regression test for C4 (running pytest must never write
# into the real analytics.db in the repo root; conftest.py's autouse
# _isolated_analytics_db fixture redirects main.analytics_db to a per-test
# temp file for the duration of the test).
# ---------------------------------------------------------------------------

def test_main_analytics_db_is_redirected_away_from_the_real_file(client):
    import main
    from analytics_db import DB_FILE

    assert main.analytics_db.db_path != DB_FILE

    real_db_exists_before = os.path.exists(DB_FILE)
    real_db_mtime_before = os.path.getmtime(DB_FILE) if real_db_exists_before else None

    client.get("/api/sample-cif-files")  # any request that goes through analytics_middleware

    real_db_exists_after = os.path.exists(DB_FILE)
    real_db_mtime_after = os.path.getmtime(DB_FILE) if real_db_exists_after else None
    assert real_db_exists_after == real_db_exists_before
    assert real_db_mtime_after == real_db_mtime_before


def test_purge_old_data_uses_utc_cutoff(db):
    """A row timestamped 2 UTC-days ago must be purged with retention_days=1,
    computed from datetime.utcnow() (not datetime.now())."""
    two_days_ago_utc = (datetime.utcnow() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S')
    _insert_access_log_at(db, two_days_ago_utc)

    deleted = db.purge_old_data(retention_days=1)
    assert deleted == 1

    with db._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM access_logs").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Review fixes: numpy params, daily-count window, recent ordering, lazy singleton
# ---------------------------------------------------------------------------

def test_log_event_accepts_numpy_parameters(db):
    import numpy as np

    db.log_event("relax", parameters={"fmax": np.float32(0.5), "n": np.int64(3),
                                      "vec": np.array([1, 2]), "obj": object})
    (ev,) = db.get_recent_events()
    assert ev["parameters"]["n"] == 3 and ev["parameters"]["vec"] == [1, 2]
    assert ev["parameters"]["fmax"] == pytest.approx(0.5)
    assert "object" in ev["parameters"]["obj"]  # unknown types fall back to str()


def test_log_event_empty_dict_is_not_null(db):
    db.log_event("relax", parameters={})
    db.log_event("relax", parameters=None)
    with db._get_connection() as conn:
        rows = conn.execute("SELECT parameters FROM analysis_events ORDER BY id").fetchall()
    assert rows == [("{}",), (None,)]


def test_daily_access_counts_exclude_rows_just_outside_window(db):
    from datetime import timezone

    now = datetime.now(timezone.utc)
    fmt = '%Y-%m-%d %H:%M:%S'
    _insert_access_log_at(db, (now - timedelta(days=1)).strftime(fmt))
    # one minute older than the 14-day window: on the boundary *date*, but outside
    _insert_access_log_at(db, (now - timedelta(days=14, minutes=1)).strftime(fmt))
    counts = db.get_daily_access_counts(days=14)
    assert sum(c["count"] for c in counts) == 1


def test_daily_access_counts_day_buckets_not_more_than_days_plus_one(db):
    from datetime import timezone

    now = datetime.now(timezone.utc)
    for d in range(30):
        _insert_access_log_at(db, (now - timedelta(days=d, minutes=1)).strftime('%Y-%m-%d %H:%M:%S'))
    assert len(db.get_daily_access_counts(days=14)) == 14


def test_recent_events_ordered_newest_first_with_id_tiebreak(db):
    same = "2030-01-01 00:00:00"
    for name in ("a", "b", "c"):
        with db._get_connection() as conn:
            conn.execute("INSERT INTO analysis_events (timestamp, event_type, filename) "
                         "VALUES (?, 'x', ?)", (same, name))
            conn.commit()
    _insert_analysis_event_at(db, "2029-01-01 00:00:00")
    assert [e["filename"] for e in db.get_recent_events()] == ["c", "b", "a", "Cu.cif"]


def test_importing_analytics_db_does_not_create_database_file(tmp_path):
    import subprocess
    import sys

    code = (
        "import analytics_db, os, sys;"
        "analytics_db.DB_FILE = sys.argv[1];"
        "assert analytics_db.analytics_db._instance is None;"
        "assert not os.path.exists(sys.argv[1]);"
    )
    target = tmp_path / "lazy.db"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-c", code, str(target)], cwd=repo, check=True)
    assert not target.exists()


def test_lazy_analytics_handle_forwards_calls(tmp_path, monkeypatch):
    import analytics_db as mod

    handle = mod._LazyAnalyticsDatabase()
    real_cls = mod.AnalyticsDatabase
    monkeypatch.setattr(mod, "AnalyticsDatabase",
                        lambda: real_cls(db_path=str(tmp_path / "x.db")))
    assert not (tmp_path / "x.db").exists()
    handle.log_event("sample_load", filename="Cu.cif")
    assert (tmp_path / "x.db").exists()
    assert handle.get_feature_usage() == [{"type": "sample_load", "count": 1}]
