"""Tests for analytics_db.AnalyticsDatabase.

Each test uses its own throwaway sqlite file (not the shared production
analytics.db) via AnalyticsDatabase(db_path=...).
"""
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
