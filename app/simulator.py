# app/simulator.py

import random
from datetime import datetime, timedelta


class DataSimulator:
    """
    Generates synthetic signals:
    - tickets
    - webhook/API logs
    - metrics (KPIs)
    Supports an injected incident:
      - stage=3 merchants get webhook 403 spike
      - payment succeeds but orders_created drops (soft failure)
      - conversion drops (business failure)
    """

    def __init__(self, seed=42):
        random.seed(seed)
        self.merchants = [f"m_{i:03d}" for i in range(1, 201)]
        self.migration_stage = {m: random.choice([1, 2, 3, 4]) for m in self.merchants}
        self._incident_on = False

        # "Business constants" for risk report
        self.default_aov = 2000  # INR (demo)

    def inject_incident(self, on: bool = True):
        self._incident_on = on

    def _now(self):
        return datetime.utcnow()

    def _rand_ts(self, within_minutes: int):
        t = self._now() - timedelta(minutes=random.randint(0, within_minutes))
        return t.isoformat(timespec="seconds") + "Z"

    def generate_window(self, window_minutes=15):
        tickets = []
        logs = []

        # Baseline volumes
        base_ticket_rate = 12
        base_log_rate = 100

        # Incident boosts
        incident_ticket_boost = 3.0

        stage3 = [m for m in self.merchants if self.migration_stage[m] == 3]
        others = [m for m in self.merchants if self.migration_stage[m] != 3]

        # Ensure stage-3 merchants are well represented
        k3 = min(70, len(stage3))
        k_other = 120 - k3
        sampled_merchants = random.sample(stage3, k=k3) + random.sample(others, k=k_other)

        # -----------------------------
        # 1) Generate Logs
        # -----------------------------
        for _ in range(base_log_rate):
            m = random.choice(sampled_merchants)
            stage = self.migration_stage[m]

            endpoint = random.choice(["/webhooks/orders", "/api/checkout", "/api/cart"])
            is_webhook = endpoint.startswith("/webhooks")

            if is_webhook:
                # webhook: mostly 200, small chance of errors
                code = random.choice([200, 200, 200, 200, 401, 403, 500])
                event_type = "webhook_success" if code == 200 else "webhook_failure"
            else:
                # api: mostly 200, some errors
                code = random.choice([200, 200, 200, 200, 200, 401, 403, 500, 504])
                event_type = "api_success" if code == 200 else "api_failure"

            # Inject incident: stage 3 webhook 403 spike
            if self._incident_on and stage == 3 and endpoint == "/webhooks/orders":
                if random.random() < 0.65:
                    code = 403
                    event_type = "webhook_failure"

            logs.append({
                "ts": self._rand_ts(window_minutes),
                "merchant_id": m,
                "event_type": event_type,
                "code": int(code),
                "endpoint": endpoint
            })

        # -----------------------------
        # 2) Generate KPI Metrics
        # -----------------------------
        # AOV fluctuates slightly
        aov = float(self.default_aov * random.uniform(0.85, 1.15))

        # Checkout attempts in this 15m window
        # (you can tune these for stronger signals)
        checkout_attempts = random.randint(180, 320)

        # Baseline success rates
        # checkout_success ~ 80-92% normally
        base_checkout_success_rate = random.uniform(0.80, 0.92)
        checkout_success = int(checkout_attempts * base_checkout_success_rate)

        # Payments succeed for a fraction of successful checkouts
        # baseline payment success rate ~ 92-98% of checkout_success
        base_payment_success_rate = random.uniform(0.92, 0.98)
        payment_success = int(checkout_success * base_payment_success_rate)

        # Orders created baseline close to payments
        # baseline: ~97-100% of payment_success
        base_order_create_rate = random.uniform(0.97, 1.00)
        orders_created = int(payment_success * base_order_create_rate)

        # Inject incident effect: payment succeeds but orders don't get created
        # (soft failure + business KPI drop)
        if self._incident_on:
            # make it dramatic but plausible:
            # a big chunk of payments don't turn into orders due to webhook/auth issue
            # e.g., orders_created becomes only 55-75% of payment_success
            soft_drop = random.uniform(0.55, 0.75)
            orders_created = int(payment_success * soft_drop)

        # Safety clamp
        orders_created = max(0, min(orders_created, payment_success))

        conversion_rate = float(orders_created / max(checkout_attempts, 1))
        checkout_success_rate = float(checkout_success / max(checkout_attempts, 1))

        # -----------------------------
        # 3) Generate Tickets
        # -----------------------------
        ticket_count = base_ticket_rate
        if self._incident_on:
            ticket_count = int(base_ticket_rate * incident_ticket_boost)

        for i in range(ticket_count):
            m = random.choice(sampled_merchants)
            stage = self.migration_stage[m]

            # Incident ticket pattern strongly tied to soft failure
            if self._incident_on and stage == 3 and random.random() < 0.70:
                subject = "Orders not created after payment"
                body = (
                    "Payment succeeds but no order.created webhook arrives. "
                    "We recently moved to headless checkout."
                )
            else:
                templates = [
                    ("Checkout broken", "Checkout fails with a generic error after headless migration."),
                    ("API auth issue", "We are seeing 401 errors when calling checkout API."),
                    ("Webhooks missing", "Our webhook endpoint is not receiving events."),
                    ("Frontend-backend mismatch", "Cart shows items but backend returns empty cart."),
                    ("Slow checkout", "Checkout is very slow and customers are abandoning."),
                ]
                subject, body = random.choice(templates)

            tickets.append({
                "ticket_id": f"t_{self._now().strftime('%H%M%S')}_{i:03d}",
                "merchant_id": m,
                "created_at": self._rand_ts(window_minutes),
                "subject": subject,
                "body": body,
                "migration_stage": stage
            })

        # -----------------------------
        # 4) Pack metrics
        # -----------------------------
        metrics = {
            "window_minutes": int(window_minutes),
            "incident_on": bool(self._incident_on),

            # stage distribution of sampled merchants
            "stage_distribution": {
                "1": sum(1 for m in sampled_merchants if self.migration_stage[m] == 1),
                "2": sum(1 for m in sampled_merchants if self.migration_stage[m] == 2),
                "3": sum(1 for m in sampled_merchants if self.migration_stage[m] == 3),
                "4": sum(1 for m in sampled_merchants if self.migration_stage[m] == 4),
            },

            # KPIs for soft/business detection + risk report
            "aov": float(round(aov, 2)),
            "checkout_attempts": int(checkout_attempts),
            "checkout_success": int(checkout_success),
            "payment_success": int(payment_success),
            "orders_created": int(orders_created),

            # Convenience rates (not required but nice for UI)
            "conversion_rate": float(round(conversion_rate, 4)),
            "checkout_success_rate": float(round(checkout_success_rate, 4)),
        }

        return tickets, logs, metrics, dict(self.migration_stage)