from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
from typing import Any, Optional

Severity = Literal["low", "medium", "high", "critical"]
ActionType = Literal[
    "support_macro",
    "proactive_merchant_message",
    "escalate_engineering",
    "docs_update_suggestion",
    "temporary_mitigation_suggestion"
]
ApprovalStatus = Literal["not_required", "pending", "approved", "rejected"]

class Ticket(BaseModel):
    ticket_id: str
    merchant_id: str
    created_at: str
    subject: str
    body: str
    migration_stage: int

class LogEvent(BaseModel):
    ts: str
    merchant_id: str
    event_type: str  # "webhook_failure", "api_failure", "checkout_failure"
    code: str        # "403", "401", "500", etc.
    endpoint: str

class ClusterSummary(BaseModel):
    cluster_id: int
    size: int
    top_terms: List[str]
    example_ticket_ids: List[str]
    representative_texts: List[str]

class Anomaly(BaseModel):
    key: str  # e.g. "stage=3|webhook_failure|403|/webhooks/orders"
    severity: Severity
    z_score: float
    current_rate: float
    baseline_rate: float
    window_minutes: int

class Observations(BaseModel):
    window_minutes: int
    tickets: List[Ticket]
    logs: List[LogEvent]
    clusters: List[ClusterSummary]
    anomalies: List[Anomaly]
    metrics: Dict[str, Any] = Field(default_factory=dict)

class ProposedAction(BaseModel):
    action_type: ActionType
    title: str
    description: str
    risk: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_approval: bool
    payload: Dict[str, Any] = Field(default_factory=dict)

class AgentAssessment(BaseModel):
    summary: str
    likely_root_cause: str
    category: Literal["merchant_misconfig", "platform_regression", "docs_gap", "migration_misstep", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str]
    assumptions: List[str]
    uncertainties: List[str]
    proposed_actions: List[ProposedAction]

class IncidentRecord(BaseModel):
    incident_id: str
    created_at: str
    status: Literal["open", "monitoring", "resolved"]
    assessment: AgentAssessment
    observations: Observations
    blast_radius_estimate: Optional[Dict[str, Any]] = None
    action_simulation: Optional[Dict[str, Any]] = None

class StoredAction(BaseModel):
    action_id: str
    incident_id: str
    created_at: str
    action: ProposedAction
    approval_status: ApprovalStatus
    approved_at: Optional[str] = None
    executed_at: Optional[str] = None
    outcome: Optional[Dict[str, Any]] = None

class Observations(BaseModel):
    window_minutes: int
    tickets: list[dict]
    logs: list[dict]
    clusters: list[dict]
    anomalies: list[dict]
    metrics: dict

    # NEW
    risk_report: dict | None = None