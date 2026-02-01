import streamlit as st
import httpx
import json

API = "http://127.0.0.1:8000"
TIMEOUT = httpx.Timeout(180.0)

# ---------------- utils ----------------
def safe_request(method: str, url: str, **kwargs):
    """Make request + safely parse JSON; show readable error on failure."""
    try:
        r = httpx.request(method, url, timeout=TIMEOUT, **kwargs)
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

    if r.status_code >= 400:
        st.error(f"Server error {r.status_code}: {(r.text or '')[:2000]}")
        return None

    txt = (r.text or "").strip()
    if not txt:
        st.error("Empty response from server (no JSON). Check server logs.")
        return None

    try:
        return r.json()
    except Exception:
        st.error("Response is not valid JSON. Showing raw text below:")
        st.code(txt[:4000], language="text")
        return None


def as_pretty_text(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def section_title(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def chip(label: str, variant: str = "neutral"):
    """
    variants: critical, high, medium, low, ok, warn, neutral
    """
    styles = {
        "critical": ("rgba(255, 74, 74, 0.16)", "rgba(255, 74, 74, 0.40)", "#ff4a4a"),
        "high":     ("rgba(255, 167, 38, 0.16)", "rgba(255, 167, 38, 0.40)", "#ffa726"),
        "medium":   ("rgba(255, 235, 59, 0.14)", "rgba(255, 235, 59, 0.35)", "#ffeb3b"),
        "low":      ("rgba(0, 230, 118, 0.14)",  "rgba(0, 230, 118, 0.35)",  "#00e676"),
        "ok":       ("rgba(0, 200, 255, 0.14)",  "rgba(0, 200, 255, 0.35)",  "#00c8ff"),
        "warn":     ("rgba(171, 71, 188, 0.14)", "rgba(171, 71, 188, 0.35)", "#ab47bc"),
        "neutral":  ("rgba(255,255,255,0.06)",   "rgba(255,255,255,0.12)",   "rgba(255,255,255,0.85)")
    }
    bg, border, fg = styles.get(variant, styles["neutral"])
    return f"""
    <span style="
        display:inline-flex;
        align-items:center;
        padding:5px 10px;
        border-radius:999px;
        background:{bg};
        border:1px solid {border};
        color:{fg};
        font-size:12px;
        font-weight:600;
        margin-right:8px;
        margin-bottom:8px;
        white-space:nowrap;
    ">{label}</span>
    """


def chip_for_severity(x: str):
    v = (x or "").strip().lower()
    if v == "critical":
        return chip("severity: critical", "critical")
    if v == "high":
        return chip("severity: high", "high")
    if v in ("medium", "med"):
        return chip("severity: medium", "medium")
    if v == "low":
        return chip("severity: low", "low")
    return chip(f"severity: {v or 'unknown'}", "neutral")


def chip_for_risk(x: str):
    v = (x or "").strip().lower()
    if v == "critical":
        return chip("risk: critical", "critical")
    if v == "high":
        return chip("risk: high", "high")
    if v in ("medium", "med"):
        return chip("risk: medium", "medium")
    if v == "low":
        return chip("risk: low", "low")
    return chip(f"risk: {v or 'unknown'}", "neutral")


def chip_for_incident(on: bool):
    return chip("incident: ON", "critical") if on else chip("incident: OFF", "low")


def card(title: str = "", subtitle: str = ""):
    """Streamlit-native card (border wrapper) — safe + full-width."""
    box = st.container(border=True)
    with box:
        if title:
            st.markdown(f"**{title}**")
        if subtitle:
            st.caption(subtitle)
    return box


def mono_block(title: str, content: str):
    st.markdown(f"**{title}**")
    st.code(content, language="text")


def _fmt_pct(x):
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return str(x)


def _fmt_money(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)


def render_risk_report(rr: dict):
    """
    Render risk report as a clean, readable report (no JSON).
    Expected shape:
      rr = { enabled, window_minutes, assumptions{...}, estimates{...}, notes }
    """
    if not rr:
        st.caption("Enable add-ons on the run or ensure backend attaches risk_report.")
        return

    enabled = rr.get("enabled", None)
    window = rr.get("window_minutes", "-")
    assumptions = rr.get("assumptions") or {}
    est = rr.get("estimates") or {}
    notes = (rr.get("notes") or "").strip()

    # Header chips
    chips = ""
    if enabled is True:
        chips += chip("enabled", "low")
    elif enabled is False:
        chips += chip("disabled", "warn")
    chips += chip(f"window: {window}m", "neutral")
    st.markdown(chips, unsafe_allow_html=True)

    # Topline loss KPIs
    loss_window = est.get("loss_in_window", 0)
    loss_hour = est.get("loss_per_hour", 0)
    loss_day = est.get("loss_per_day", 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"Loss / {window} min", _fmt_money(loss_window))
    with c2:
        st.metric("Loss / hour", _fmt_money(loss_hour))
    with c3:
        st.metric("Loss / day", _fmt_money(loss_day))

    # Volume / expectation
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        st.metric("Attempts", est.get("attempts", "-"))
    with c5:
        st.metric("Orders created", est.get("orders_created", "-"))
    with c6:
        st.metric("Expected orders", est.get("expected_orders", "-"))
    with c7:
        st.metric("Lost orders", est.get("lost_orders", "-"))

    st.markdown("---")

    # Assumptions
    st.markdown("#### Assumptions")
    aov = assumptions.get("aov", None)
    bcr = assumptions.get("baseline_conversion_rate", None)
    ccr = assumptions.get("current_conversion_rate", None)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("AOV", _fmt_money(aov) if aov is not None else "-")
    with a2:
        st.metric("Baseline conversion", _fmt_pct(bcr) if bcr is not None else "-")
    with a3:
        st.metric("Current conversion", _fmt_pct(ccr) if ccr is not None else "-")

    if notes:
        st.markdown("#### Notes")
        st.write(notes)


def _get_rag_context(incident: dict):
    """
    Supports BOTH formats:
    1) incident["rag_context"] as a string (your current payload)
    2) incident["rag_context_used"] as list of dict hits (older format)
    Also checks under observations just in case.
    """
    if not incident:
        return None

    if incident.get("rag_context"):
        return incident["rag_context"]

    if incident.get("rag_context_used"):
        return incident["rag_context_used"]

    obs = incident.get("observations") or {}
    if obs.get("rag_context"):
        return obs["rag_context"]
    if obs.get("rag_context_used"):
        return obs["rag_context_used"]

    return None


# ---------------- Page ----------------
st.set_page_config(page_title="Agentic Support MVP", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Page spacing */
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
div[data-testid="stSidebarContent"] { padding-top: 1rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
code { border-radius: 10px !important; }
.smallmuted { opacity: 0.72; font-size: 12px; }

/* ✅ Turn Streamlit border containers into “cards” */
div[data-testid="stVerticalBlockBorderWrapper"]{
  width: 100% !important;
  box-sizing: border-box !important;
  border-radius: 16px !important;
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  padding: 14px 14px 12px 14px !important;
  margin-bottom: 14px !important;
}

/* Remove extra inner padding that Streamlit adds */
div[data-testid="stVerticalBlockBorderWrapper"] > div{
  padding: 0 !important;
}

/* Reduce extra paragraph spacing inside markdown */
div[data-testid="stMarkdownContainer"] > p { margin-bottom: 0.35rem; }
</style>
""", unsafe_allow_html=True)

st.title("🛠️ Agentic AI: Self-Healing Support Layer")
st.caption("Operational Intelligence + Guardrails + RAG (docs/runbooks) + Optional LLM narrative")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 📚 Knowledge Base (RAG)")
    st.caption("Upload runbooks/docs so the agent can cite them.")

    colS1, colS2 = st.columns(2)
    with colS1:
        if st.button("RAG Stats", use_container_width=True):
            stats = safe_request("GET", f"{API}/rag/stats")
            if stats:
                st.session_state["rag_stats"] = stats
    with colS2:
        if st.button("Clear RAG", use_container_width=True):
            cleared = safe_request("POST", f"{API}/rag/clear")
            if cleared:
                st.success("Cleared RAG store.")

    if "rag_stats" in st.session_state:
        st.code(as_pretty_text(st.session_state["rag_stats"]), language="text")

    uploaded = st.file_uploader(
        "Upload docs (txt / md / pdf)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True
    )
    if st.button("Upload to RAG", use_container_width=True):
        if not uploaded:
            st.warning("Please select at least one file.")
        else:
            files_payload = [("files", (f.name, f.getvalue())) for f in uploaded]
            res = safe_request("POST", f"{API}/rag/upload", files=files_payload)
            if res:
                st.success(f"Added chunks: {res.get('added_chunks', 0)}")
                st.caption(f"Sources: {', '.join(res.get('sources', []))}")

    st.divider()
    st.markdown("## ⚙️ Demo controls")
    colI1, colI2 = st.columns(2)
    with colI1:
        if st.button("Incident ON", use_container_width=True):
            safe_request("POST", f"{API}/inject_incident", params={"on": True})
            st.success("Incident enabled.")
    with colI2:
        if st.button("Incident OFF", use_container_width=True):
            safe_request("POST", f"{API}/inject_incident", params={"on": False})
            st.info("Incident disabled.")

    window_minutes = st.slider("Observation window (minutes)", 5, 60, 15, step=5)
    enable_addons = st.toggle("Enable add-ons (blast radius + risk)", value=True)
    use_llm = st.toggle("Use LLM (Ollama)", value=False)

# ---------------- Top actions ----------------
t1, t2, t3 = st.columns([1, 1, 1])
with t1:
    if st.button("👀 Observe", use_container_width=True):
        obs = safe_request("GET", f"{API}/observe", params={"window_minutes": window_minutes})
        if obs:
            st.session_state["last_obs"] = obs
            st.toast("Observation captured.", icon="✅")

with t2:
    if st.button("🔁 Run Agent Cycle", use_container_width=True):
        cyc = safe_request(
            "POST", f"{API}/run_cycle",
            params={"window_minutes": window_minutes, "enable_addons": enable_addons, "use_llm": use_llm}
        )
        if cyc:
            st.session_state["last_cycle"] = cyc
            st.toast("Agent cycle complete.", icon="✅")

with t3:
    if st.button("🧹 Clear local UI state", use_container_width=True):
        for k in ["last_obs", "last_cycle", "rag_stats"]:
            st.session_state.pop(k, None)
        st.toast("Cleared UI state.", icon="🧽")

st.divider()

# ---------------- Main layout ----------------
left, right = st.columns([1.2, 0.8], gap="large")

# ========== LEFT ==========
with left:
    section_title("Latest Observation / Cycle", "Detailed report view (human readable)")

    obs = st.session_state.get("last_obs")
    if obs:
        metrics = obs.get("metrics", {}) or {}
        incident_on = bool(metrics.get("incident_on", False))

        pills = chip(f"window: {obs.get('window_minutes', window_minutes)}m", "neutral")
        pills += chip_for_incident(incident_on)
        if "checkout_attempts" in metrics:
            pills += chip(f"attempts: {metrics.get('checkout_attempts')}", "neutral")
        if "conversion_rate" in metrics:
            pills += chip(f"conversion: {_fmt_pct(metrics.get('conversion_rate'))}", "ok")
        if "aov" in metrics:
            pills += chip(f"AOV: {_fmt_money(metrics.get('aov'))}", "neutral")

        st.markdown(pills, unsafe_allow_html=True)
        st.markdown("---")

        # Clusters
        clusters = obs.get("clusters", [])[:5]
        section_title("Ticket clusters", "Top clusters with representative examples")
        if not clusters:
            with card("No clusters formed yet."):
                st.caption("Run a few cycles / increase window if needed.")
        else:
            for c in clusters[:5]:
                with card(f"Cluster #{c.get('cluster_id')}  •  size {c.get('size')}"):
                    st.caption(f"Top terms: {', '.join(c.get('top_terms', []))}")
                    example = (c.get("representative_texts") or ["-"])[0]
                    st.write(example)

        # Anomalies
        anomalies = obs.get("anomalies", [])[:10]
        section_title("Anomalies", "Hard / Soft / Business failures with severity & risk")
        if not anomalies:
            with card("No anomalies detected in this window."):
                st.caption("Looks healthy for this observation window.")
        else:
            for a in anomalies[:10]:
                key = a.get("key", "unknown")
                sev = a.get("severity", "unknown")
                risk = a.get("risk", "unknown")
                btype = a.get("broken_type", "unknown")
                cat = a.get("category", "unknown")

                chips = chip(f"type: {btype}", "neutral")
                chips += chip_for_severity(sev)
                chips += chip_for_risk(risk)
                chips += chip(f"category: {cat}", "neutral")

                with card(key):
                    st.markdown(chips, unsafe_allow_html=True)

                    extra_bits = []
                    if a.get("current_rate") is not None:
                        extra_bits.append(f"rate={a.get('current_rate')}/min")
                    if a.get("baseline_rate") is not None:
                        extra_bits.append(f"baseline={a.get('baseline_rate')}/min")
                    if a.get("z_score") is not None:
                        extra_bits.append(f"z={a.get('z_score')}")
                    if a.get("velocity") is not None:
                        extra_bits.append(f"velocity={a.get('velocity')}")

                    if extra_bits:
                        st.caption(" • ".join(extra_bits))

                    if a.get("notes"):
                        st.write(a.get("notes"))

        with st.expander("Raw observation (text)"):
            st.code(as_pretty_text(obs), language="text")
    else:
        with card("No observation yet."):
            st.write("Click **Observe** to generate one window of signals (tickets/logs/metrics).")

    # Cycle / Incident / Assessment
    cyc = st.session_state.get("last_cycle")
    if cyc:
        inc = (cyc.get("incident") or {})
        assessment = (inc.get("assessment") or {})
        stored_actions = inc.get("stored_actions") or cyc.get("stored_actions") or []

        st.markdown("---")
        section_title("Agent assessment", "Root cause + confidence + evidence + uncertainties + actions")

        summary = assessment.get("summary", "-")
        root = assessment.get("likely_root_cause", "-")
        category = assessment.get("category", "-")
        conf = assessment.get("confidence", "-")

        with card("Assessment"):
            st.write(f"**Summary:** {summary}")
            st.write(f"**Likely root cause:** {root}")
            st.write(f"**Category:** {category}")
            st.write(f"**Confidence:** {conf}")

        # ✅ RAG CONTEXT
        st.markdown("---")
        section_title("RAG context used", "What the agent cited from your uploaded docs")
        rag_ctx = _get_rag_context(inc)

        if isinstance(rag_ctx, str) and rag_ctx.strip():
            with card("Top retrieved context"):
                st.code(rag_ctx, language="text")
        elif isinstance(rag_ctx, list) and rag_ctx:
            for h in rag_ctx[:5]:
                src = h.get("source", "unknown")
                cid = h.get("chunk_id", "")
                dist = h.get("distance", "")
                txt = (h.get("text") or "").strip()
                if len(txt) > 1500:
                    txt = txt[:1500] + "…"
                mono_block(f"[{src}] {cid} dist={dist}", txt)
        else:
            with card("No RAG context returned"):
                st.caption("RAG may be empty, not uploaded yet, or not attached for this run.")

        # Evidence / Uncertainties / Assumptions
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        ev = assessment.get("evidence", []) or []
        un = assessment.get("uncertainties", []) or []
        asm = assessment.get("assumptions", []) or []

        with c1:
            with card("Evidence"):
                if ev:
                    for x in ev:
                        st.write(f"• {x}")
                else:
                    st.caption("None.")

        with c2:
            with card("Uncertainties"):
                if un:
                    for x in un:
                        st.write(f"• {x}")
                else:
                    st.caption("None.")

        with c3:
            with card("Assumptions"):
                if asm:
                    for x in asm:
                        st.write(f"• {x}")
                else:
                    st.caption("None.")

        # ✅ Proposed actions
        actions = (assessment.get("proposed_actions") or [])
        st.markdown("---")
        section_title("Proposed actions", "Auto-run allowed actions; approve gated ones in the Approvals panel")

        if not actions:
            with card("No proposed actions returned."):
                st.caption("Agent didn’t propose actions for this incident.")
        else:
            for a in actions:
                atype = a.get("action_type", "unknown")
                title = a.get("title", "")
                desc = a.get("description", "")
                risk = a.get("risk", "unknown")
                req = bool(a.get("requires_human_approval", False))
                aconf = a.get("confidence", "-")

                chips = chip(f"type: {atype}", "neutral")
                chips += chip_for_risk(risk)
                chips += chip(f"conf: {aconf}", "neutral")
                chips += chip("approval: required", "warn") if req else chip("approval: not required", "low")

                with card(title):
                    st.markdown(chips, unsafe_allow_html=True)
                    st.caption(desc)

        # ✅ Stored actions (DB)
        if stored_actions:
            st.markdown("---")
            section_title("Stored actions (DB)", "These are the action_ids saved; approvals + execution uses these IDs")
            for it in stored_actions[:25]:
                a = it.get("action", {}) or {}
                aid = it.get("action_id", "")
                status = it.get("approval_status", "unknown")
                atype = a.get("action_type", "unknown")
                title = a.get("title", "")
                rrisk = a.get("risk", "unknown")
                req = bool(a.get("requires_human_approval", False))

                chips = chip(f"id: {aid}", "neutral")
                chips += chip(f"status: {status}", "ok" if status != "pending" else "warn")
                chips += chip(f"type: {atype}", "neutral")
                chips += chip_for_risk(rrisk)
                chips += chip("approval: required", "warn") if req else chip("approval: not required", "low")

                with card(title):
                    st.markdown(chips, unsafe_allow_html=True)

        # ✅ Risk report (human readable)
        risk_report = (inc.get("observations") or {}).get("risk_report") or inc.get("risk_report")
        st.markdown("---")
        section_title("Risk report", "Revenue loss estimate based on conversion drop")
        with card("Risk report"):
            render_risk_report(risk_report)

        # ✅ Blast radius
        br = inc.get("blast_radius_estimate")
        if br:
            st.markdown("---")
            section_title("Blast radius", "Estimated scope of affected merchants/stages")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Likely impacted stage", br.get("likely_impacted_stage", "-"))
            with col2:
                st.metric("At-risk merchants", br.get("estimated_at_risk_merchants", "-"))
            with col3:
                st.metric("Confidence", br.get("confidence", "-"))
            with st.expander("Blast radius (text)"):
                st.code(as_pretty_text(br), language="text")

        with st.expander("Raw incident (text)"):
            st.code(as_pretty_text(inc), language="text")
    else:
        with card("No agent cycle yet."):
            st.write("Click **Run Agent Cycle** to generate incident + assessment + actions + add-ons.")

# ========== RIGHT ==========
with right:
    section_title("Approvals & Actions", "Approve high-risk actions; execute after approval (simulated)")

    pending = safe_request("GET", f"{API}/actions", params={"only_pending": True})
    if pending is None:
        with card("Could not load pending actions."):
            st.caption("Server might be restarting.")
    elif not pending:
        with card("No pending actions right now."):
            st.caption("If actions require approval, they will appear here.")
    else:
        for item in pending[:15]:
            a = item.get("action", {}) or {}
            action_id = item.get("action_id", "")
            atype = a.get("action_type", "unknown")
            title = a.get("title", "")
            desc = a.get("description", "")
            risk = a.get("risk", "unknown")
            conf = a.get("confidence", "unknown")

            chips = chip(f"type: {atype}", "neutral")
            chips += chip_for_risk(risk)
            chips += chip(f"conf: {conf}", "neutral")

            with card(title):
                st.markdown(chips, unsafe_allow_html=True)
                st.caption(f"Action ID: {action_id}")
                st.write(desc)

                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("Approve", key=f"ap_{action_id}", use_container_width=True):
                        safe_request("POST", f"{API}/approve_action", params={"action_id": action_id, "approve": True})
                        st.toast("Approved.", icon="✅")
                with c2:
                    if st.button("Reject", key=f"rej_{action_id}", use_container_width=True):
                        safe_request("POST", f"{API}/approve_action", params={"action_id": action_id, "approve": False})
                        st.toast("Rejected.", icon="🛑")
                with c3:
                    if st.button("Execute", key=f"ex_{action_id}", use_container_width=True):
                        safe_request("POST", f"{API}/execute_action", params={"action_id": action_id})
                        st.toast("Executed.", icon="⚡")

    st.markdown("---")
    section_title("Recent incidents (audit trail)", "Expand any incident for details")

    incs = safe_request("GET", f"{API}/incidents")
    if incs:
        for row in incs[:10]:
            iid = row.get("incident_id", "")
            status = row.get("status", "")
            created = row.get("created_at", "")
            with st.expander(f"{iid} • {status} • {created}"):
                incobj = row.get("incident", {}) or {}

                st.markdown("**Assessment (text)**")
                st.code(as_pretty_text(incobj.get("assessment", {})), language="text")

                rr = (incobj.get("observations") or {}).get("risk_report") or incobj.get("risk_report")
                if rr:
                    st.markdown("**Risk report (text)**")
                    st.code(as_pretty_text(rr), language="text")

                br = incobj.get("blast_radius_estimate")
                if br:
                    st.markdown("**Blast radius (text)**")
                    st.code(as_pretty_text(br), language="text")

                rag_ctx = _get_rag_context(incobj)
                if rag_ctx:
                    st.markdown("**RAG context (text)**")
                    if isinstance(rag_ctx, str):
                        st.code(rag_ctx, language="text")
                    else:
                        st.code(as_pretty_text(rag_ctx[:5]), language="text")

                st.markdown("**Raw incident (text)**")
                st.code(as_pretty_text(incobj), language="text")
    else:
        with card("No incidents yet."):
            st.caption("Run an agent cycle to generate incidents.")