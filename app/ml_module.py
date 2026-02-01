# app/ml_module.py

import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer


class OperationalIntelligence:
    """
    Operational Intelligence (ML/DL module)

    1) Ticket clustering:
       - SentenceTransformer embeddings + DBSCAN (cosine)

    2) Hard failures (log-based):
       - spike detection per (stage, event_type, code, endpoint)
       - hybrid detectors: z-score + EWMA + CUSUM + IsolationForest
       - structured anomaly object: category, broken_type, risk, severity, velocity

    3) Soft failures (multi-signal / KPI gaps):
       - payment success but orders not created (gap)

    4) Business failures (KPI trend):
       - conversion drop / checkout success drop (EWMA + CUSUM)

    5) Risk report:
       - estimate revenue loss based on baseline conversion vs current conversion
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        ewma_alpha: float = 0.35,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        iforest_contamination: float = 0.05,
        iforest_min_points: int = 40,
        iforest_refit_every: int = 15,
    ):
        # Embeddings + cache for speed
        self.embedder = SentenceTransformer(model_name)
        self._embed_cache = {}

        # --- Time-series storage for HARD failures (per key) ---
        self._rate_hist = defaultdict(list)      # key -> list[rate]
        self._ewma_state = {}                    # key -> ewma value
        self._cusum_state = {}                   # key -> {"pos": float, "neg": float}

        # --- Time-series storage for KPI series (SOFT/BUSINESS) ---
        self._kpi_hist = defaultdict(list)       # metric_key -> list[value]
        self._kpi_ewma_state = {}                # metric_key -> ewma
        self._kpi_cusum_state = {}               # metric_key -> {"pos": float, "neg": float}

        # Detector parameters
        self.ewma_alpha = float(ewma_alpha)
        self.cusum_k = float(cusum_k)
        self.cusum_h = float(cusum_h)

        # IsolationForest (global model on feature vectors)
        self._iforest = None
        self._iforest_X = []                     # list of feature vectors
        self._iforest_last_fit_n = 0
        self.iforest_contamination = float(iforest_contamination)
        self.iforest_min_points = int(iforest_min_points)
        self.iforest_refit_every = int(iforest_refit_every)

    # -------------------- 1) Ticket clustering --------------------

    def cluster_tickets(self, tickets, max_examples: int = 3, cap: int = 40):
        if not tickets:
            return []

        tickets = tickets[:cap]
        texts = [f"{t.get('subject', '')} :: {t.get('body', '')}" for t in tickets]

        # Cached embeddings
        emb_list = [None] * len(texts)
        to_encode, to_encode_idx = [], []

        for i, tx in enumerate(texts):
            if tx in self._embed_cache:
                emb_list[i] = self._embed_cache[tx]
            else:
                to_encode.append(tx)
                to_encode_idx.append(i)

        if to_encode:
            new_emb = self.embedder.encode(to_encode, normalize_embeddings=True)
            for j, idx in enumerate(to_encode_idx):
                self._embed_cache[texts[idx]] = new_emb[j]
                emb_list[idx] = new_emb[j]

        emb = np.array(emb_list)

        clustering = DBSCAN(eps=0.25, min_samples=3, metric="cosine")
        labels = clustering.fit_predict(emb)

        clusters = {}
        for idx, lbl in enumerate(labels):
            if lbl == -1:
                continue
            clusters.setdefault(lbl, []).append(idx)

        summaries = []
        for cid, idxs in clusters.items():
            tokens = []
            for i in idxs:
                tokens += [w.lower().strip(".,:;()[]{}<>!?\"'") for w in texts[i].split()]
            common = [w for w, _ in Counter(tokens).most_common(12) if len(w) > 3][:5]

            example_ids = [tickets[i].get("ticket_id") for i in idxs[:max_examples]]
            reps = [texts[i] for i in idxs[:max_examples]]

            summaries.append({
                "cluster_id": int(cid),
                "size": int(len(idxs)),
                "top_terms": common,
                "example_ticket_ids": example_ids,
                "representative_texts": reps
            })

        summaries.sort(key=lambda x: x["size"], reverse=True)
        return summaries
    
    def estimate_blast_radius(self, anomalies, tickets, merchant_stage_map):
        """
        Add-on: rough 'blast radius' estimator.
        For MVP: estimate likely stage impacted and how many merchants could be affected.
        Never throws (safe).
        """
        try:
            if not anomalies:
                return {
                    "likely_impacted_stage": None,
                    "estimated_at_risk_merchants": 0,
                    "confidence": 0.2,
                    "notes": "No anomalies; blast radius not estimated."
                }

            # infer stage from anomaly keys like 'stage=3|...'
            implied_stages = []
            for a in anomalies[:8]:
                key = a.get("key", "")
                for s in (1, 2, 3, 4):
                    if f"stage={s}" in key:
                        implied_stages.append(s)

            stage = max(implied_stages) if implied_stages else None

            at_risk = set()
            if stage is not None:
                for m, st in merchant_stage_map.items():
                    if st == stage:
                        at_risk.add(m)

            # confidence heuristic
            ticket_count = len(tickets)
            top_sev = anomalies[0].get("severity", "low")
            sev_bonus = 0.35 if top_sev == "critical" else (0.2 if top_sev == "high" else 0.05)
            confidence = min(0.95, 0.35 + ticket_count / 60.0 + sev_bonus)

            return {
                "likely_impacted_stage": stage,
                "estimated_at_risk_merchants": len(at_risk),
                "confidence": round(confidence, 2),
                "notes": "Heuristic estimate for MVP; refine with merchant segmentation later."
            }

        except Exception as e:
            return {
                "likely_impacted_stage": None,
                "estimated_at_risk_merchants": 0,
                "confidence": 0.1,
                "notes": f"Blast radius estimation failed safely: {type(e).__name__}: {e}"
            }

    # -------------------- helpers: categorization & severity --------------------

    @staticmethod
    def _endpoint_criticality(endpoint: str) -> str:
        ep = (endpoint or "").lower()
        if "checkout" in ep or "payment" in ep:
            return "critical"
        if "cart" in ep or "orders" in ep:
            return "high"
        if "webhooks" in ep:
            return "high"
        return "medium"

    @staticmethod
    def _categorize_hard(event_type: str, code: int, endpoint: str) -> tuple[str, str]:
        """
        Returns (category, broken_type)
        broken_type for log-based is typically "hard".
        """
        ep = (endpoint or "").lower()
        et = (event_type or "").lower()
        c = int(code) if code is not None else 0

        if "webhooks" in ep or "webhook" in et:
            # webhook-specific failures are hard failures
            if c in (401, 403):
                return "webhook_auth_error", "hard"
            if c >= 500:
                return "webhook_downstream_error", "hard"
            return "webhook_delivery_error", "hard"

        if c in (401, 403):
            return "auth_error", "hard"
        if c == 429:
            return "rate_limit", "hard"
        if c in (408, 504):
            return "timeout", "hard"
        if 500 <= c <= 599:
            return "server_error", "hard"
        if 400 <= c <= 499:
            return "client_error", "hard"

        return "unknown", "hard"

    @staticmethod
    def _base_risk_from_category(category: str, endpoint: str) -> str:
        crit = OperationalIntelligence._endpoint_criticality(endpoint)
        if crit == "critical":
            return "critical"
        if crit == "high":
            return "high"
        if category in ("server_error", "timeout", "webhook_downstream_error"):
            return "high"
        if category in ("auth_error", "rate_limit", "client_error", "webhook_auth_error"):
            return "medium"
        return "low"

    @staticmethod
    def _velocity(history: list[float], current: float, lookback: int = 4) -> float:
        """
        Relative velocity = (current - mean(prev)) / max(mean(prev), eps)
        """
        eps = 1e-6
        if len(history) < 2:
            return 0.0
        prev = history[-lookback:] if len(history) >= lookback else history[:-1]
        if not prev:
            return 0.0
        base = float(np.mean(prev))
        return float((current - base) / max(base, eps))

    @staticmethod
    def _velocity_severity(velocity: float, base_sev: str) -> str:
        """
        Upgrade severity based on how fast it's increasing.
        velocity is relative change.
        Example:
          +1% -> +10% quickly => velocity ~ 9.0 (huge) => critical
        """
        # Gentle: keep base
        if velocity < 0.3:
            return base_sev

        # Moderate acceleration: at least "high"
        if velocity < 1.0:
            return "high" if base_sev in ("low", "medium") else base_sev

        # Fast acceleration: critical
        return "critical"

    @staticmethod
    def _severity_from_scores(z: float, ewma_resid_z: float, cusum_pos: float, cusum_neg: float) -> str:
        """
        Combine different detectors into base severity.
        """
        score = max(abs(z), abs(ewma_resid_z), abs(cusum_pos), abs(cusum_neg))
        if score >= 8:
            return "critical"
        if score >= 4:
            return "high"
        if score >= 2:
            return "medium"
        return "low"

    # -------------------- IsolationForest (global) --------------------

    def _maybe_fit_iforest(self):
        n = len(self._iforest_X)
        if n < self.iforest_min_points:
            return
        # Refit every few points to keep cost low
        if (n - self._iforest_last_fit_n) < self.iforest_refit_every and self._iforest is not None:
            return

        X = np.array(self._iforest_X, dtype=float)
        self._iforest = IsolationForest(
            n_estimators=150,
            contamination=self.iforest_contamination,
            random_state=42
        )
        self._iforest.fit(X)
        self._iforest_last_fit_n = n

    def _iforest_is_anomalous(self, x_vec: list[float]) -> tuple[bool, float]:
        """
        Returns (is_anom, score). Score is the decision_function output (higher is more normal).
        """
        if self._iforest is None:
            return False, 0.0
        X = np.array([x_vec], dtype=float)
        score = float(self._iforest.decision_function(X)[0])
        pred = int(self._iforest.predict(X)[0])  # -1 anomaly, 1 normal
        return (pred == -1), score

    # -------------------- 2) HARD failures detection --------------------

    def detect_hard_anomalies(self, logs, merchant_stage_map, window_minutes: int = 15):
        if not logs:
            return []

        counts = Counter()
        meta = {}  # key -> (stage, event_type, code, endpoint)
        for e in logs:
            stage = int(merchant_stage_map.get(e.get("merchant_id"), 0))
            event_type = e.get("event_type", "unknown")
            code = int(e.get("code", 0))
            endpoint = e.get("endpoint", "unknown")
            key = f"stage={stage}|{event_type}|{code}|{endpoint}"

            counts[key] += 1
            meta[key] = (stage, event_type, code, endpoint)

        anomalies = []
        for key, c in counts.items():
            stage, event_type, code, endpoint = meta[key]
            if code >= 200 and code < 300:
                continue
            if c < 2:
                continue
            current_rate = float(c / max(window_minutes, 1))

            history = self._rate_hist[key]
            mu = float(np.mean(history)) if history else current_rate
            sd = float(max(np.std(history), 0.01)) if len(history) >= 5 else 0.1  # safe

            z = float((current_rate - mu) / sd) if len(history) >= 5 else 0.0

            # EWMA
            prev_ewma = self._ewma_state.get(key, mu)
            ewma = float(self.ewma_alpha * current_rate + (1 - self.ewma_alpha) * prev_ewma)
            self._ewma_state[key] = ewma
            ewma_resid_z = float((ewma - mu) / sd) if len(history) >= 5 else 0.0

            # CUSUM on standardized residual
            st = self._cusum_state.get(key, {"pos": 0.0, "neg": 0.0})
            resid_z = float((current_rate - mu) / sd) if len(history) >= 5 else 0.0
            st["pos"] = max(0.0, st["pos"] + resid_z - self.cusum_k)
            st["neg"] = min(0.0, st["neg"] + resid_z + self.cusum_k)
            self._cusum_state[key] = st

            # Velocity (relative change)
            vel = self._velocity(history, current_rate, lookback=4)

            # IsolationForest features (global): keep it simple & robust
            # [current_rate, mu, z, ewma, vel, code_bucket, endpoint_crit_bucket]
            code_bucket = 2.0 if 500 <= code <= 599 else (1.0 if 400 <= code <= 499 else 0.0)
            crit = self._endpoint_criticality(endpoint)
            crit_bucket = {"critical": 3.0, "high": 2.0, "medium": 1.0, "low": 0.0}.get(crit, 1.0)

            x_vec = [current_rate, mu, z, ewma, vel, code_bucket, crit_bucket]
            self._iforest_X.append(x_vec)
            if len(self._iforest_X) > 400:
                self._iforest_X.pop(0)
            self._maybe_fit_iforest()
            if_anom, if_score = self._iforest_is_anomalous(x_vec)

            # Update history after computing scores
            history.append(current_rate)
            if len(history) > 60:
                history.pop(0)

            # Decide anomaly: any strong detector triggers
            cusum_trigger = (st["pos"] >= self.cusum_h) or (abs(st["neg"]) >= self.cusum_h)
            z_trigger = (len(history) >= 5 and z >= 3.0)
            ewma_trigger = (len(history) >= 5 and ewma_resid_z >= 3.0)
            if_trigger = if_anom

            if not (cusum_trigger or z_trigger or ewma_trigger or if_trigger):
                continue

            category, broken_type = self._categorize_hard(event_type, code, endpoint)
            risk = self._base_risk_from_category(category, endpoint)

            base_sev = self._severity_from_scores(z, ewma_resid_z, st["pos"], st["neg"])
            severity = self._velocity_severity(vel, base_sev)

            anomalies.append({
                "key": key,
                "broken_type": broken_type,     # "hard"
                "category": category,
                "risk": risk,
                "severity": severity,

                "stage": stage,
                "event_type": event_type,
                "code": code,
                "endpoint": endpoint,

                "count": int(c),
                "window_minutes": int(window_minutes),
                "current_rate": float(round(current_rate, 4)),
                "baseline_rate": float(round(mu, 4)),

                "z_score": float(round(z, 2)),
                "ewma": float(round(ewma, 4)),
                "ewma_resid_z": float(round(ewma_resid_z, 2)),
                "cusum_pos": float(round(st["pos"], 2)),
                "cusum_neg": float(round(st["neg"], 2)),
                "velocity": float(round(vel, 3)),

                "iforest_anom": bool(if_anom),
                "iforest_score": float(round(if_score, 3)),
            })

        # Sort: severity first, then z-score
        sev_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        anomalies.sort(key=lambda a: (sev_rank.get(a["severity"], 0), a.get("z_score", 0)), reverse=True)
        return anomalies

    # -------------------- 3) SOFT & BUSINESS failures (KPI-based) --------------------

    def _kpi_ewma_cusum(self, metric_key: str, value: float):
        """
        Returns a dict containing ewma, cusum_pos/neg, baseline(mean), sd, velocity, triggers.
        """
        hist = self._kpi_hist[metric_key]
        mu = float(np.mean(hist)) if hist else value
        sd = float(max(np.std(hist), 0.01)) if len(hist) >= 5 else 0.1

        # EWMA
        prev_ewma = self._kpi_ewma_state.get(metric_key, mu)
        ewma = float(self.ewma_alpha * value + (1 - self.ewma_alpha) * prev_ewma)
        self._kpi_ewma_state[metric_key] = ewma
        ewma_resid_z = float((ewma - mu) / sd) if len(hist) >= 5 else 0.0

        # CUSUM
        st = self._kpi_cusum_state.get(metric_key, {"pos": 0.0, "neg": 0.0})
        resid_z = float((value - mu) / sd) if len(hist) >= 5 else 0.0
        st["pos"] = max(0.0, st["pos"] + resid_z - self.cusum_k)
        st["neg"] = min(0.0, st["neg"] + resid_z + self.cusum_k)
        self._kpi_cusum_state[metric_key] = st

        vel = self._velocity(hist, value, lookback=4)

        # Update history
        hist.append(float(value))
        if len(hist) > 80:
            hist.pop(0)

        triggers = {
            "ewma_trigger": (len(hist) >= 5 and ewma_resid_z >= 3.0),
            "cusum_trigger": (st["pos"] >= self.cusum_h) or (abs(st["neg"]) >= self.cusum_h),
        }

        return {
            "mu": mu,
            "sd": sd,
            "ewma": ewma,
            "ewma_resid_z": ewma_resid_z,
            "cusum_pos": st["pos"],
            "cusum_neg": st["neg"],
            "velocity": vel,
            **triggers
        }

    def detect_soft_and_business_failures(self, metrics: dict, window_minutes: int = 15):
        """
        Uses KPI metrics to detect:
          - Soft failures: payment success but orders not created
          - Business failures: conversion drop / checkout success drop

        Expects (if available):
          metrics["checkout_attempts"]
          metrics["checkout_success"]
          metrics["payment_success"]
          metrics["orders_created"]
          metrics["aov"]  (average order value)
        """
        if not metrics:
            return []

        attempts = float(metrics.get("checkout_attempts", 0) or 0)
        checkout_success = float(metrics.get("checkout_success", 0) or 0)
        payment_success = float(metrics.get("payment_success", 0) or 0)
        orders_created = float(metrics.get("orders_created", 0) or 0)

        # Avoid divide by zero
        eps = 1e-6
        conv_rate = float(orders_created / max(attempts, eps))
        checkout_success_rate = float(checkout_success / max(attempts, eps))

        # Soft failure proxy: payment succeeded but order not created
        # gap is fraction of attempts that turned into paid but not ordered
        soft_gap = float((payment_success - orders_created) / max(attempts, eps))

        findings = []

        # --- Soft failure detection ---
        if attempts > 20:  # avoid noisy tiny samples
            k = "soft_gap_payment_minus_order"
            s = self._kpi_ewma_cusum(k, soft_gap)

            # Soft issues are often "hidden": we trigger at lower levels too
            # Also allow absolute threshold
            abs_trigger = soft_gap >= 0.03  # 3% gap is suspicious
            trigger = s["ewma_trigger"] or s["cusum_trigger"] or abs_trigger

            if trigger:
                base_sev = self._severity_from_scores(
                    z=0.0,
                    ewma_resid_z=s["ewma_resid_z"],
                    cusum_pos=s["cusum_pos"],
                    cusum_neg=s["cusum_neg"]
                )
                severity = self._velocity_severity(s["velocity"], base_sev)

                findings.append({
                    "key": "soft_failure|payment_success_but_no_order",
                    "broken_type": "soft",
                    "category": "integration_gap",
                    "risk": "high",
                    "severity": severity,

                    "window_minutes": int(window_minutes),
                    "attempts": int(attempts),
                    "payment_success": int(payment_success),
                    "orders_created": int(orders_created),

                    "current_value": float(round(soft_gap, 4)),
                    "baseline_value": float(round(s["mu"], 4)),
                    "ewma": float(round(s["ewma"], 4)),
                    "ewma_resid_z": float(round(s["ewma_resid_z"], 2)),
                    "cusum_pos": float(round(s["cusum_pos"], 2)),
                    "cusum_neg": float(round(s["cusum_neg"], 2)),
                    "velocity": float(round(s["velocity"], 3)),

                    "notes": "Soft failure: payments succeed but order creation lags/fails (webhooks/API mismatch)."
                })

        # --- Business failure detection: conversion drop ---
        if attempts > 20:
            k = "business_conversion_rate"
            s = self._kpi_ewma_cusum(k, conv_rate)

            # For business failures we care about DROPS too:
            # If ewma is far BELOW baseline, use negative cusum or relative drop.
            rel_drop = (s["mu"] - conv_rate) / max(s["mu"], eps) if s["mu"] > 0 else 0.0
            drop_trigger = (rel_drop >= 0.15)  # 15% conversion drop
            trigger = s["cusum_trigger"] or drop_trigger  # keep it simple

            if trigger:
                # Severity increases with bigger drop and velocity
                base_sev = "medium"
                if rel_drop >= 0.30:
                    base_sev = "high"
                if rel_drop >= 0.50:
                    base_sev = "critical"
                severity = self._velocity_severity(s["velocity"], base_sev)

                findings.append({
                    "key": "business_failure|conversion_drop",
                    "broken_type": "business",
                    "category": "kpi_degradation",
                    "risk": "critical",
                    "severity": severity,

                    "window_minutes": int(window_minutes),
                    "attempts": int(attempts),
                    "orders_created": int(orders_created),

                    "current_value": float(round(conv_rate, 4)),
                    "baseline_value": float(round(s["mu"], 4)),
                    "relative_drop": float(round(rel_drop, 3)),
                    "ewma": float(round(s["ewma"], 4)),
                    "cusum_pos": float(round(s["cusum_pos"], 2)),
                    "cusum_neg": float(round(s["cusum_neg"], 2)),
                    "velocity": float(round(s["velocity"], 3)),

                    "notes": "Business failure: conversion rate dropped meaningfully vs baseline."
                })

        # --- Business failure detection: checkout success drop (optional) ---
        if attempts > 20:
            k = "business_checkout_success_rate"
            s = self._kpi_ewma_cusum(k, checkout_success_rate)
            rel_drop = (s["mu"] - checkout_success_rate) / max(s["mu"], eps) if s["mu"] > 0 else 0.0
            drop_trigger = (rel_drop >= 0.20)  # 20% success drop
            trigger = s["cusum_trigger"] or drop_trigger

            if trigger:
                base_sev = "medium"
                if rel_drop >= 0.35:
                    base_sev = "high"
                if rel_drop >= 0.60:
                    base_sev = "critical"
                severity = self._velocity_severity(s["velocity"], base_sev)

                findings.append({
                    "key": "business_failure|checkout_success_drop",
                    "broken_type": "business",
                    "category": "kpi_degradation",
                    "risk": "critical",
                    "severity": severity,

                    "window_minutes": int(window_minutes),
                    "attempts": int(attempts),
                    "checkout_success": int(checkout_success),

                    "current_value": float(round(checkout_success_rate, 4)),
                    "baseline_value": float(round(s["mu"], 4)),
                    "relative_drop": float(round(rel_drop, 3)),
                    "ewma": float(round(s["ewma"], 4)),
                    "cusum_pos": float(round(s["cusum_pos"], 2)),
                    "cusum_neg": float(round(s["cusum_neg"], 2)),
                    "velocity": float(round(s["velocity"], 3)),

                    "notes": "Business failure: checkout success rate dropped vs baseline."
                })

        # Sort by severity
        sev_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        findings.sort(key=lambda a: sev_rank.get(a["severity"], 0), reverse=True)
        return findings

    # -------------------- 4) Risk report --------------------

    def risk_report(self, metrics: dict, window_minutes: int = 15) -> dict:
        """
        Better version:
          - baseline conversion rate from historical KPI series
          - expected orders this window = baseline_conv * attempts
          - lost orders = max(0, expected - actual)
          - revenue loss = lost_orders * AOV
        """
        if not metrics:
            return {"enabled": False, "reason": "No metrics provided."}

        attempts = float(metrics.get("checkout_attempts", 0) or 0)
        orders_created = float(metrics.get("orders_created", 0) or 0)
        aov = float(metrics.get("aov", 2000) or 2000)  # default AOV for demo

        eps = 1e-6
        current_conv = float(orders_created / max(attempts, eps))

        # Baseline conversion from KPI history if available
        hist = self._kpi_hist.get("business_conversion_rate", [])
        baseline_conv = float(np.mean(hist)) if len(hist) >= 5 else current_conv

        expected_orders = float(baseline_conv * attempts)
        lost_orders = float(max(0.0, expected_orders - orders_created))

        loss_15m = float(lost_orders * aov)
        per_hour = float(loss_15m * (60.0 / max(window_minutes, 1)))
        per_day = float(per_hour * 24.0)

        return {
            "enabled": True,
            "window_minutes": int(window_minutes),
            "assumptions": {
                "aov": float(round(aov, 2)),
                "baseline_conversion_rate": float(round(baseline_conv, 4)),
                "current_conversion_rate": float(round(current_conv, 4)),
            },
            "estimates": {
                "attempts": int(attempts),
                "orders_created": int(orders_created),
                "expected_orders": float(round(expected_orders, 2)),
                "lost_orders": float(round(lost_orders, 2)),
                "loss_in_window": float(round(loss_15m, 2)),
                "loss_per_hour": float(round(per_hour, 2)),
                "loss_per_day": float(round(per_day, 2)),
            },
            "notes": "MVP loss model. In production, incorporate recovery/retry probability and segment by merchant/region/device."
        }

    # -------------------- MAIN entry: combine all failures --------------------

    def detect_anomalies(self, logs, merchant_stage_map, window_minutes: int = 15, metrics: dict | None = None):
        """
        Returns a combined list of anomalies across:
          - hard failures (log-based)
          - soft failures (KPI gap)
          - business failures (KPI trend)
        """
        hard = self.detect_hard_anomalies(logs, merchant_stage_map, window_minutes=window_minutes)
        soft_business = self.detect_soft_and_business_failures(metrics or {}, window_minutes=window_minutes)
        combined = hard + soft_business

        sev_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        combined.sort(key=lambda a: (sev_rank.get(a.get("severity", "low"), 0), a.get("z_score", 0)), reverse=True)
        return combined