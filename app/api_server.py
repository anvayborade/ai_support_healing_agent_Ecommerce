from fastapi import FastAPI
from pydantic import TypeAdapter
from datetime import datetime

from .storage import (
    init_db, init_learning_priors,
    list_incidents, list_actions,
    set_action_approval, set_action_executed,
    get_incident
)
from .schemas import Observations
from .simulator import DataSimulator
from .ml_module import OperationalIntelligence
from .agent_core import run_agent_cycle
from .config import settings

from fastapi import FastAPI, UploadFile, File
from typing import List
from .rag_store import RagStore

app = FastAPI(title="Agentic Support MVP (Ollama)")

OBS_ADAPTER = TypeAdapter(Observations)

sim = DataSimulator()
oi = OperationalIntelligence()
rag = RagStore(persist_dir="rag_db", collection_name="support_kb")

@app.get("/rag/stats")
def rag_stats():
    return rag.stats()

@app.post("/rag/clear")
def rag_clear():
    return rag.clear()

@app.post("/rag/upload")
async def rag_upload(files: List[UploadFile] = File(...)):
    payload = []
    for f in files:
        b = await f.read()
        payload.append({"filename": f.filename, "bytes": b})
    return rag.add_documents(payload)

@app.get("/rag/query")
def rag_query(q: str, k: int = 4):
    hits = rag.query(q, k=k)
    return [{"source": h.source, "chunk_id": h.chunk_id, "distance": h.distance, "text": h.text} for h in hits]

@app.on_event("startup")
def _startup():
    init_db()
    init_learning_priors()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/inject_incident")
def inject_incident(on: bool = True):
    sim.inject_incident(on)
    return {"incident_on": on}

@app.get("/observe")
def observe(window_minutes: int = 15):
    tickets, logs, metrics, stage_map = sim.generate_window(window_minutes=window_minutes)
    clusters = oi.cluster_tickets(tickets)
    anomalies = oi.detect_anomalies(logs, stage_map, window_minutes=window_minutes, metrics=metrics)

    risk = oi.risk_report(metrics, window_minutes=window_minutes)

    obs = {
        "window_minutes": window_minutes,
        "tickets": tickets,
        "logs": logs,
        "clusters": clusters,
        "anomalies": anomalies,
        "metrics": metrics,
        "risk_report":risk
    }
    obs_obj = OBS_ADAPTER.validate_python(obs)
    return obs_obj.model_dump()

@app.post("/run_cycle")
def run_cycle(window_minutes: int = 15, enable_addons: bool = False, use_llm: bool=True):
    tickets, logs, metrics, stage_map = sim.generate_window(window_minutes=window_minutes)
    clusters = oi.cluster_tickets(tickets)
    anomalies = oi.detect_anomalies(logs, stage_map, window_minutes=window_minutes, metrics=metrics)

    risk = oi.risk_report(metrics, window_minutes=window_minutes)

    obs = Observations(
        window_minutes=window_minutes,
        tickets=tickets,
        logs=logs,
        clusters=clusters,
        anomalies=anomalies,
        metrics=metrics,
        risk_report=risk
    )

    blast = None
    sim_out = None
    if enable_addons:
        blast = oi.estimate_blast_radius(anomalies, tickets, stage_map)
        sim_out = {
            "note": "Action simulation v1 placeholder. We'll upgrade later.",
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z"
        }

    orig = settings.use_llm
    settings.use_llm = use_llm

    # --- RAG retrieval: build a query from signals ---
    top_keys = [a.get("key","") for a in anomalies[:3]]
    top_terms = []
    if clusters:
        top_terms = clusters[0].get("top_terms", [])[:5]

    rag_q = f"headless checkout migration issues. anomalies: {top_keys}. cluster_terms: {top_terms}"
    hits = rag.query(rag_q, k=4)

    rag_context = ""
    if hits:
        blocks = []
        for h in hits:
            blocks.append(f"[{h.source} | {h.chunk_id} | dist={h.distance:.3f}]\n{h.text}")
        rag_context = "\n\n---\n\n".join(blocks)
    
    try:
        incident_id, incident, stored_actions = run_agent_cycle(
            obs,
            blast_radius=blast,
            action_simulation=sim_out,
            rag_context=rag_context
        )
    finally:
        settings.use_llm = orig

    return {"incident_id": incident_id, "incident": incident, "stored_actions": stored_actions}

@app.get("/incidents")
def incidents():
    return list_incidents()

@app.get("/actions")
def actions(incident_id: str | None = None, only_pending: bool = False):
    return list_actions(incident_id=incident_id, only_pending=only_pending)

@app.post("/approve_action")
def approve_action(action_id: str, approve: bool):
    status = "approved" if approve else "rejected"
    set_action_approval(action_id, status)
    return {"action_id": action_id, "status": status}

@app.post("/execute_action")
def execute_action(action_id: str):
    outcome = {
        "executed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "expected_improvement": 0.18,
        "notes": "Executed post-approval (simulated)."
    }
    set_action_executed(action_id, outcome)
    return {"action_id": action_id, "outcome": outcome}

@app.get("/incident/{incident_id}")
def get_inc(incident_id: str):
    inc = get_incident(incident_id)
    return inc if inc else {"error": "not found"}