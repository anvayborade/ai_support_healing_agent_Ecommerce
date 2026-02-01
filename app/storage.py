import sqlite3
import json
from datetime import datetime
from .config import settings

def _now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def init_db():
    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS incidents(
        incident_id TEXT PRIMARY KEY,
        created_at TEXT,
        status TEXT,
        incident_json TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS actions(
        action_id TEXT PRIMARY KEY,
        incident_id TEXT,
        created_at TEXT,
        approval_status TEXT,
        approved_at TEXT,
        executed_at TEXT,
        action_json TEXT,
        outcome_json TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS learning(
        action_type TEXT PRIMARY KEY,
        alpha REAL,
        beta REAL
    )
    """)

    conn.commit()
    conn.close()

def upsert_incident(incident_id: str, incident_obj: dict, status: str):
    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO incidents(incident_id, created_at, status, incident_json)
    VALUES(?,?,?,?)
    ON CONFLICT(incident_id) DO UPDATE SET
      status=excluded.status,
      incident_json=excluded.incident_json
    """, (incident_id, _now(), status, json.dumps(incident_obj)))
    conn.commit()
    conn.close()

def list_incidents(limit: int = 50):
    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()
    cur.execute("SELECT incident_id, created_at, status, incident_json FROM incidents ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({"incident_id": r[0], "created_at": r[1], "status": r[2], "incident": json.loads(r[3])})
    return out

def get_incident(incident_id: str):
    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()
    cur.execute("SELECT incident_json FROM incidents WHERE incident_id=?", (incident_id,))
    row = cur.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None

def insert_action(action_id: str, incident_id: str, action_obj: dict, approval_status: str):
    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO actions(action_id, incident_id, created_at, approval_status, approved_at, executed_at, action_json, outcome_json)
    VALUES(?,?,?,?,?,?,?,?)
    """, (action_id, incident_id, _now(), approval_status, None, None, json.dumps(action_obj), None))
    conn.commit()
    conn.close()

def list_actions(incident_id: str | None = None, only_pending: bool = False):
    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()
    q = "SELECT action_id, incident_id, created_at, approval_status, approved_at, executed_at, action_json, outcome_json FROM actions"
    params = []
    clauses = []
    if incident_id:
        clauses.append("incident_id=?")
        params.append(incident_id)
    if only_pending:
        clauses.append("approval_status='pending'")
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY created_at DESC"
    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "action_id": r[0],
            "incident_id": r[1],
            "created_at": r[2],
            "approval_status": r[3],
            "approved_at": r[4],
            "executed_at": r[5],
            "action": json.loads(r[6]),
            "outcome": json.loads(r[7]) if r[7] else None
        })
    return out

def set_action_approval(action_id: str, status: str):
    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()
    cur.execute("""
    UPDATE actions SET approval_status=?, approved_at=? WHERE action_id=?
    """, (status, _now(), action_id))
    conn.commit()
    conn.close()

def set_action_executed(action_id: str, outcome_obj: dict):
    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()
    cur.execute("""
    UPDATE actions SET executed_at=?, outcome_json=? WHERE action_id=?
    """, (_now(), json.dumps(outcome_obj), action_id))
    conn.commit()
    conn.close()

def init_learning_priors(default_alpha=1.0, default_beta=1.0):
    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM learning")
    n = cur.fetchone()[0]
    if n == 0:
        action_types = [
            "support_macro",
            "proactive_merchant_message",
            "escalate_engineering",
            "docs_update_suggestion",
            "temporary_mitigation_suggestion"
        ]
        for a in action_types:
            cur.execute("INSERT INTO learning(action_type, alpha, beta) VALUES(?,?,?)",
                        (a, default_alpha, default_beta))
        conn.commit()
    conn.close()

def get_learning_params():
    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()
    cur.execute("SELECT action_type, alpha, beta FROM learning")
    rows = cur.fetchall()
    conn.close()
    return {r[0]: {"alpha": r[1], "beta": r[2]} for r in rows}

def update_learning(action_type: str, success: bool):
    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()
    cur.execute("SELECT alpha, beta FROM learning WHERE action_type=?", (action_type,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return
    alpha, beta = row
    if success:
        alpha += 1.0
    else:
        beta += 1.0
    cur.execute("UPDATE learning SET alpha=?, beta=? WHERE action_type=?", (alpha, beta, action_type))
    conn.commit()
    conn.close()