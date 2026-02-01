from uuid import uuid4
from datetime import datetime

def _now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

RISKY_ACTIONS = {"proactive_merchant_message", "temporary_mitigation_suggestion"}

def requires_approval(action_type: str) -> bool:
    return action_type in RISKY_ACTIONS

def simulate_execute(action_obj: dict, current_context: dict) -> dict:
    a_type = action_obj["action_type"]
    base_effect = {
        "support_macro": 0.10,
        "docs_update_suggestion": 0.05,
        "escalate_engineering": 0.12,
        "proactive_merchant_message": 0.18,
        "temporary_mitigation_suggestion": 0.25
    }.get(a_type, 0.05)

    severity = current_context.get("top_anomaly_severity", "low")
    severity_multiplier = {"low": 0.7, "medium": 1.0, "high": 1.2, "critical": 1.35}.get(severity, 1.0)

    expected_improvement = base_effect * severity_multiplier
    return {
        "executed_at": _now(),
        "expected_improvement": round(expected_improvement, 3),
        "notes": "Simulated outcome. Real system would measure metrics pre/post."
    }

def new_id(prefix: str):
    return f"{prefix}_{uuid4().hex[:10]}"