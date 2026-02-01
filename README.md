# Agentic Support MVP — Self-Healing Support Layer (LLM + ML + Time-Series + RAG)

An end-to-end **agentic incident intelligence system** that monitors operational signals (tickets, logs, KPIs), detects anomalies, clusters issues, retrieves relevant runbook context via **RAG**, and proposes (and optionally stores/executes) corrective actions — all surfaced in a clean **Streamlit UI** and served by a **FastAPI backend**.

This project is built around the agent loop:

> **observe → reason → decide → act**

---

## 🚩 Problem Statement

During complex platform migrations (e.g., **headless checkout migration**), merchants experience:
- Slow checkout / abandonment
- Frontend–backend mismatches (cart empty on backend)
- API auth failures (401/403)
- Webhook failures (403/500) causing **payment success but missing order.created**
- Conversion drops leading to significant revenue loss

Traditional monitoring shows “errors”, but not:
- **What is the most likely root cause?**
- **Which merchants/stages are affected?**
- **What is the estimated business impact (loss)?**
- **What actions should we take right now?**
- **Which runbook section supports the decision?**

This system answers all of the above in one place.

---

## ✨ What Makes This Stand Out

### ✅ Hybrid Intelligence Architecture (LLM + ML + Time-Series)
- **LLM (Llama 3.2 via Ollama)** generates structured assessments and action narratives (optional toggle).
- **ML Clustering (DBSCAN)** groups similar support tickets into human-readable clusters with top terms + representative examples.
- **DL / Time-Series Detection** flags hard failures, soft failures, and KPI degradations using time-series signals and anomaly scoring.
- **RAG (Retrieval-Augmented Generation)** cites relevant runbook chunks to ground decisions in documented reality.
- **Auto Actions + Approval Workflow** stores proposed actions, supports “approval required” gating, and allows simulated execution.

### ✅ Business-First: Risk Report
- Converts conversion drops into:
  - lost orders
  - estimated revenue loss per window/hour/day
- Keeps the report human-readable in UI (not raw JSON)

---

## 🧠 Core Capabilities

### 1) Observe (Signal Ingestion)
Ingests a rolling window of:
- Support tickets (subjects + bodies + migration stage)
- Logs (API/webhook success/failure codes, endpoints)
- Aggregated metrics (attempts, conversion, checkout success, AOV, etc.)

### 2) Reason (Intelligence Layer)
- Clusters ticket themes (DBSCAN)
- Detects anomalies:
  - **Hard** failures (e.g., webhook 403 spikes)
  - **Soft** failures (payment success but no order created)
  - **Business** failures (conversion drop)

### 3) Decide (Assessment + Impact + Actions)
Produces an incident assessment:
- Summary
- Likely root cause + category + confidence
- Evidence + assumptions + uncertainties
- Blast radius estimate (impacted stage, at-risk merchants)
- Risk report (revenue loss estimate)
- Proposed actions (with risk + confidence + approval requirements)

### 4) Act (Action System)
- Stores suggested actions with IDs (DB / in-memory)
- Approval workflow:
  - Approve / reject / execute (simulated)
- Supports safe automation patterns:
  - Auto-execute low-risk actions
  - Require human approval for high-risk actions

---

## 🧩 System Components

### Backend — FastAPI
- `/observe` — generate observations (windowed)
- `/run_cycle` — full agent loop (observe→reason→decide→act)
- `/incidents` — audit trail
- `/actions`, `/approve_action`, `/execute_action` — approvals + action execution
- RAG endpoints:
  - `/rag/upload` — upload runbooks/docs
  - `/rag/stats` — index stats
  - `/rag/clear` — clear store

### UI — Streamlit
- Upload docs into RAG
- Trigger Observe / Run Agent Cycle
- Visualize:
  - ticket clusters
  - anomaly list with severity + risk chips
  - agent assessment
  - RAG context used
  - risk report dashboard
  - stored actions + approvals panel
  - incident audit trail

### RAG System
- Ingests MD/TXT/PDF runbooks
- Splits into chunks
- Vector retrieval returns top chunks per incident
- Output attaches into incident payload as `rag_context`

### LLM Layer (Optional)
- Uses **Ollama + Llama 3.2**
- When enabled, adds a clearer narrative for:
  - summaries
  - root cause explanation
  - action descriptions

---

### To run:

- Start server in one terminal:
    - uvicorn app.api_server:app --reload

- Start streamlit in second terminal:
    - streamlit run app/ui_app.py