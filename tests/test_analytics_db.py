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
