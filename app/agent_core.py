from uuid import uuid4
from datetime import datetime
from pydantic import TypeAdapter
from ollama import Client

from .schemas import Observations, AgentAssessment
from .config import settings
from .learning import thompson_rank_actions, record_outcome
from .actions import requires_approval, simulate_execute, new_id
from .storage import upsert_incident, insert_action, set_action_executed

def _now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

ASSESSMENT_ADAPTER = TypeAdapter(AgentAssessment)

SYSTEM_PROMPT = """You are an agentic support operations coordinator for a SaaS e-commerce platform migrating hosted -> headless.
You must be explainable and cautious. Money & live checkouts are high risk.

Given OBSERVATIONS (clusters, anomalies, tickets, logs), produce:
- summary
- likely_root_cause
- category (merchant_misconfig/platform_regression/docs_gap/migration_misstep/unknown)
- confidence 0..1
- evidence (list of strings)
- assumptions (list of strings)
- uncertainties (list of strings)
- proposed_actions: list with:
  - action_type
  - title
  - description
  - risk (low/medium/high/critical)
  - confidence 0..1
  - requires_human_approval (true if risky)
  - payload (structured hints)

Constraints:
- If evidence is weak, reduce confidence and propose gathering more signals.
- Never propose auto-changing checkout/security rules without human approval.
- Prefer safe actions first: support_macro, escalate_engineering, docs_update_suggestion.
Return VALID JSON matching the schema exactly (no extra keys).

Calibration rules:
- If evidence list has fewer than 3 concrete points, confidence MUST be <= 0.70.
- If category is "unknown", confidence MUST be <= 0.60.
- Evidence must cite anomalies/clusters/metrics (not just restating the ticket).
- Always propose at least 2 actions: one safe action + one optional action.
- "proactive_merchant_message" risk must be "high" (not critical) unless money-loss is proven by metrics.
"""
def compact_observations(obs: dict) -> dict:
    tickets = obs.get("tickets", [])
    logs = obs.get("logs", [])
    clusters = obs.get("clusters", [])
    anomalies = obs.get("anomalies", [])
    metrics = obs.get("metrics", {})

    # keep only a few example tickets
    sample_tickets = []
    for t in tickets[:5]:
        sample_tickets.append({
            "ticket_id": t["ticket_id"],
            "merchant_id": t["merchant_id"],
            "migration_stage": t["migration_stage"],
            "subject": t["subject"],
            "body": (t["body"][:200] if t.get("body") else "")
        })

    # summarize logs into top patterns (instead of sending every log line)
    log_summary = {}
    for e in logs:
        k = f"{e['event_type']}|{e['code']}|{e['endpoint']}"
        log_summary[k] = log_summary.get(k, 0) + 1
    top_log_summary = sorted(log_summary.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "window_minutes": obs.get("window_minutes"),
        "metrics": metrics,
        "ticket_count": len(tickets),
        "log_count": len(logs),
        "clusters_top": clusters[:3],
        "anomalies_top": anomalies[:5],
        "sample_tickets": sample_tickets,
        "top_log_summary": top_log_summary,
    }

def llm_assess(observations: dict, rag_context: str | None = None) -> dict:
    # Support both full and compact payloads
    anoms = observations.get("anomalies") or observations.get("anomalies_top") or []
    clusters = observations.get("clusters") or observations.get("clusters_top") or []
    top_anom = anoms[0] if anoms else None
    top_cluster = clusters[0] if clusters else None

    # FAST fallback (no LLM)
    if not settings.use_llm:
        # If RAG context contains useful hints, use it in a simple way
        rag_hint = (rag_context or "").lower()
        if "signature" in rag_hint and "403" in (top_anom.get("key","") if top_anom else ""):
            guess = "Webhook signature/header mismatch (docs indicate signature requirements changed)."
            category = "docs_gap"
            conf = 0.8
        else:
            guess = "Webhook endpoint not receiving events" if top_anom and "webhook" in top_anom.get("key","") else "Unknown"
            category = "unknown"
            conf = 0.55 if guess != "Unknown" else 0.35

        return {
            "summary": "Detected clusters/anomalies; generated a likely cause (fast mode).",
            "likely_root_cause": guess,
            "category": category,
            "confidence": conf,
            "evidence": [
                f"Top anomaly: {top_anom['key']}" if top_anom else "No strong anomalies",
                f"Top cluster: {top_cluster['top_terms']}" if top_cluster else "No clusters"
            ],
            "assumptions": ["Signals are representative of current merchant issues."],
            "uncertainties": ["Need confirmation from logs/docs whether this is platform regression vs merchant misconfig."],
            "proposed_actions": []
        }

    # LLM path (Ollama) – include retrieved context
    client = Client(host=settings.ollama_host)
    schema = AgentAssessment.model_json_schema()

    rag_block = ""
    if rag_context:
        rag_block = f"\n\nRAG_CONTEXT (from internal docs/runbooks):\n{rag_context}\n"

    response = client.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + "\nUse RAG_CONTEXT if provided. Cite it in evidence when relevant."},
            {"role": "user", "content": f"OBSERVATIONS:\n{observations}{rag_block}"}
        ],
        format=schema,
        stream=False
    )

    content = response["message"]["content"]
    parsed = ASSESSMENT_ADAPTER.validate_json(content)
    return parsed.model_dump()

def propose_default_actions(assessment: dict, ranked_action_types: list[tuple[str, float]]):
    if assessment.get("proposed_actions") and len(assessment["proposed_actions"]) >= 2:
        return assessment

    top = ranked_action_types[:3]
    actions = []
    for a_type, score in top:
        title_map = {
            "support_macro": "Draft support macro for the dominant ticket cluster",
            "proactive_merchant_message": "Prepare proactive message for at-risk merchants",
            "escalate_engineering": "Create engineering escalation with evidence bundle",
            "docs_update_suggestion": "Draft docs update suggestion for missing/unclear steps",
            "temporary_mitigation_suggestion": "Suggest temporary mitigation (requires approval)"
        }
        actions.append({
            "action_type": a_type,
            "title": title_map.get(a_type, a_type),
            "description": "Auto-generated based on current learning priority. Refine with evidence if needed.",
            "risk": "high" if a_type in ["proactive_merchant_message", "temporary_mitigation_suggestion"] else "medium",
            "confidence": float(min(0.8, 0.55 + score/2)),
            "requires_human_approval": requires_approval(a_type),
            "payload": {}
        })
    assessment["proposed_actions"] = actions
    return assessment

def run_agent_cycle(observations: Observations, blast_radius: dict | None = None, action_simulation: dict | None = None, rag_context: str | None = None):
    obs_dict = observations.model_dump()
    incident_id = f"inc_{uuid4().hex[:10]}"

    ranked = thompson_rank_actions()
    assessment = llm_assess(obs_dict, rag_context=rag_context)   # <-- changed
    assessment = propose_default_actions(assessment, ranked)

    for a in assessment["proposed_actions"]:
        a["requires_human_approval"] = requires_approval(a["action_type"])

    incident = {
        "incident_id": incident_id,
        "created_at": _now(),
        "status": "open",
        "assessment": assessment,
        "observations": obs_dict,
        "rag_context": rag_context,  # <-- store for audit/debug/demo
        "blast_radius_estimate": blast_radius,
        "action_simulation": action_simulation
    }

    # Store actions & execute safe ones immediately (simulated)
    top_sev = obs_dict["anomalies"][0]["severity"] if obs_dict["anomalies"] else "low"
    context = {"top_anomaly_severity": top_sev}

    stored_actions = []
    for a in assessment["proposed_actions"]:
        action_id = new_id("act")
        approval_status = "pending" if a["requires_human_approval"] else "not_required"
        insert_action(action_id, incident_id, a, approval_status)

        if not a["requires_human_approval"]:
            outcome = simulate_execute(a, context)
            set_action_executed(action_id, outcome)

            success = outcome.get("expected_improvement", 0) >= 0.12
            record_outcome(a["action_type"], success)

        stored_actions.append({"action_id": action_id, "approval_status": approval_status, "action": a})

    # ✅ attach to incident for audit trail
    incident["stored_actions"] = stored_actions

    # ✅ now store the final incident (after actions exist)
    upsert_incident(incident_id, incident, status="open")

    return incident_id, incident, stored_actions