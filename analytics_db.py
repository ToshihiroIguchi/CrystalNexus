
import sqlite3
import logging
from datetime import datetime, timedelta
import json
import os
from contextlib import contextmanager

# Use relative path for database file
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics.db")

# Setup logger if not already configured in main app
logger = logging.getLogger("CrystalNexus.Analytics")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class AnalyticsDatabase:
    def __init__(self, db_path=DB_FILE):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Table for general access logs (middleware)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS access_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        path TEXT,
                        method TEXT,
                        status_code INTEGER,
                        client_host TEXT,
                        user_agent TEXT
                    )
                """)
                
                # Table for specific analysis events
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        event_type TEXT,
                        filename TEXT,
                        formula TEXT,
                        num_atoms INTEGER,
                        execution_time REAL,
                        parameters TEXT,
                        session_id TEXT
                    )
                """)
                
                conn.commit()
                logger.info(f"Analytics database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize analytics database: {e}")

    def log_access(self, path: str, method: str, status_code: int, client_host: str, user_agent: str):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO access_logs (path, method, status_code, client_host, user_agent) 
                    VALUES (?, ?, ?, ?, ?)
                """, (path, method, status_code, client_host, user_agent))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log access: {e}")

    def log_event(self, event_type: str, filename: str = None, formula: str = None, 
                  num_atoms: int = None, execution_time: float = None, 
                  parameters: dict = None, session_id: str = None):
        try:
            params_json = json.dumps(parameters) if parameters else None
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO analysis_events 
                    (event_type, filename, formula, num_atoms, execution_time, parameters, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (event_type, filename, formula, num_atoms, execution_time, params_json, session_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    # --- Aggregation Methods for Dashboard ---

    def get_daily_access_counts(self, days=7):
        """Get daily access counts for the last N days."""
        try:
            date_threshold = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT date(timestamp), COUNT(*) 
                    FROM access_logs 
                    WHERE timestamp >= ? 
                    GROUP BY date(timestamp) 
                    ORDER BY date(timestamp)
                """, (date_threshold,))
                return [{"date": row[0], "count": row[1]} for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get daily access counts: {e}")
            return []
    
    def get_feature_usage(self):
        """Get usage counts by feature type (event_type)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT event_type, COUNT(*) 
                    FROM analysis_events 
                    GROUP BY event_type
                    ORDER BY COUNT(*) DESC
                """)
                return [{"type": row[0], "count": row[1]} for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get feature usage: {e}")
            return []

    def get_popular_samples(self, limit=10):
        """Get most popular sample files loaded."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT filename, COUNT(*) as count 
                    FROM analysis_events 
                    WHERE event_type = 'sample_load' 
                    GROUP BY filename 
                    ORDER BY count DESC 
                    LIMIT ?
                """, (limit,))
                return [{"name": row[0], "count": row[1]} for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get popular samples: {e}")
            return []
            
    def get_recent_events(self, limit=20):
        """Get recent analysis events."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        id, timestamp, event_type, filename, formula, 
                        num_atoms, execution_time, parameters
                    FROM analysis_events 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                columns = ["id", "timestamp", "event_type", "filename", "formula", 
                           "num_atoms", "execution_time", "parameters"]
                
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(zip(columns, row))
                    # Parse parameters JSON if present
                    if row_dict["parameters"]:
                        try:
                            row_dict["parameters"] = json.loads(row_dict["parameters"])
                        except:
                            pass
                    results.append(row_dict)
                return results
        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []

# Singleton instance
analytics_db = AnalyticsDatabase()
