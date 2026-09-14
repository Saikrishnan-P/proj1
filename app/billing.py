"""
Usage-based billing for the third-party API platform: prepaid credits,
topped up via Stripe Checkout (one-time payments, not subscriptions or
Stripe's metered-billing API). A customer buys a credit pack, the
balance is deducted per /ingest or /query call, and 402 is returned once
it hits zero -- never a surprise invoice, and never negative.

Flow:
    1. POST /developers/billing/checkout (see main.py) creates a Stripe
       Checkout Session for a chosen pack and returns its URL. The
       chosen pack's credit count travels in the session's metadata so
       fulfillment doesn't need a second lookup.
    2. The customer pays on Stripe's hosted page.
    3. Stripe POSTs checkout.session.completed to
       POST /webhooks/stripe (see main.py), signed with
       STRIPE_WEBHOOK_SECRET. handle_stripe_webhook() verifies the
       signature and calls fulfill_checkout_session(), which credits
       the account -- idempotently, since Stripe can and does redeliver
       webhooks (see app/db.py's idx_credit_tx_reference).
    4. app/main.py's get_principal() dependency calls deduct_credits()
       before doing the actual work for an API-key-authenticated
       request, so a customer can never go below zero mid-call.
"""
from __future__ import annotations

import time
import uuid

from app.config import settings
from app.db import get_cursor

# Example packs -- tune freely. Each is a one-time Stripe Checkout
# purchase (no subscription), identified by `id` in the checkout
# request. price_cents is USD cents; credits is what's granted on
# successful payment. Kept as plain price_data on the Checkout Session
# (see create_checkout_session) rather than pre-created Stripe Price
# objects, so there's nothing to set up in the Stripe dashboard beyond
# an API key.
CREDIT_PACKS: dict[str, dict] = {
    "starter": {"name": "Starter", "credits": 1_000, "price_cents": 1_000},   # $10 / 1,000 credits
    "growth":  {"name": "Growth",  "credits": 6_000, "price_cents": 5_000},   # $50 / 6,000 credits
    "scale":   {"name": "Scale",   "credits": 15_000, "price_cents": 10_000}, # $100 / 15,000 credits
}


class InsufficientCreditsError(Exception):
    """Raised by deduct_credits when balance < amount. main.py maps this
    to a 402 Payment Required."""


class BillingError(Exception):
    """Raised for a malformed checkout/webhook request; main.py maps
    this to a 400."""


# ---------------------------------------------------------------------------
# Costs -- see app/config.py for the underlying settings and why each
# call type is priced this way.
# ---------------------------------------------------------------------------

def query_credit_cost() -> int:
    return settings.query_credit_cost


def ingest_credit_cost(chunks_indexed: int) -> int:
    """Scales with repo size (real embedding compute) but stays a flat
    multiple of a fixed chunk bucket, not literally 1 credit/chunk, so a
    customer can estimate cost from repo size alone before ingesting."""
    per_bucket = max(1, settings.ingest_credit_cost_per_chunks)
    return max(settings.ingest_credit_cost_minimum, -(-chunks_indexed // per_bucket))  # ceil div


# ---------------------------------------------------------------------------
# Credit ledger
# ---------------------------------------------------------------------------

def get_balance(api_customer_id: str) -> int:
    with get_cursor(dict_rows=False) as cur:
        cur.execute("SELECT credit_balance FROM api_customers WHERE id = %s", (api_customer_id,))
        row = cur.fetchone()
    if row is None:
        raise BillingError(f"No such API customer '{api_customer_id}'.")
    return row[0]


def deduct_credits(api_customer_id: str, amount: int, reason: str, allow_negative: bool = False) -> int:
    """Atomically checks-and-deducts `amount` credits, raising
    InsufficientCreditsError instead of ever letting the balance go
    negative. SELECT ... FOR UPDATE takes a row lock for the duration of
    this transaction so two concurrent requests from the same customer
    can't both read the same starting balance and both succeed when only
    one should. Returns the new balance.

    allow_negative=True is for the ONE case where the compute has
    already happened before the true cost is known -- /ingest, whose
    cost depends on chunks_indexed, only found out once the background
    job finishes (see app/jobs.py's run_ingest_job). main.py still does
    an upfront balance>0 check before enqueueing to catch the common
    "no credits at all" case early, but a large repo can legitimately
    cost more than the pre-check implied; refusing to record that
    actual usage would just make the ledger wrong. Ordinary calls
    (/query) know their flat cost before doing any work and should
    never pass this.
    """
    if amount <= 0:
        return get_balance(api_customer_id)

    with get_cursor(dict_rows=False) as cur:
        cur.execute(
            "SELECT credit_balance FROM api_customers WHERE id = %s FOR UPDATE",
            (api_customer_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise BillingError(f"No such API customer '{api_customer_id}'.")
        balance = row[0]
        if balance < amount and not allow_negative:
            raise InsufficientCreditsError(
                f"Insufficient credits: {balance} available, {amount} required."
            )

        new_balance = balance - amount
        cur.execute(
            "UPDATE api_customers SET credit_balance = %s WHERE id = %s",
            (new_balance, api_customer_id),
        )
        cur.execute(
            "INSERT INTO credit_transactions (id, api_customer_id, delta, reason, reference, created_at) "
            "VALUES (%s, %s, %s, %s, NULL, %s)",
            (uuid.uuid4().hex, api_customer_id, -amount, reason, time.time()),
        )
    return new_balance


def grant_credits(api_customer_id: str, amount: int, reason: str, reference: str | None = None) -> int:
    """Adds credits (a purchase fulfillment or a manual/admin grant).
    `reference` is how fulfill_checkout_session() below makes a Stripe
    session's grant idempotent -- see idx_credit_tx_reference in
    app/db.py. Returns the new balance."""
    with get_cursor(dict_rows=False) as cur:
        cur.execute(
            "SELECT credit_balance FROM api_customers WHERE id = %s FOR UPDATE",
            (api_customer_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise BillingError(f"No such API customer '{api_customer_id}'.")

        if reference is not None:
            cur.execute(
                "SELECT 1 FROM credit_transactions WHERE reference = %s", (reference,)
            )
            if cur.fetchone() is not None:
                # Already fulfilled -- a Stripe webhook redelivery.
                # Return the current balance unchanged rather than
                # crediting the same purchase twice.
                return row[0]

        new_balance = row[0] + amount
        cur.execute(
            "UPDATE api_customers SET credit_balance = %s WHERE id = %s",
            (new_balance, api_customer_id),
        )
        cur.execute(
            "INSERT INTO credit_transactions (id, api_customer_id, delta, reason, reference, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (uuid.uuid4().hex, api_customer_id, amount, reason, reference, time.time()),
        )
    return new_balance


def get_recent_transactions(api_customer_id: str, limit: int = 50) -> list[dict]:
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, delta, reason, created_at FROM credit_transactions "
            "WHERE api_customer_id = %s ORDER BY created_at DESC LIMIT %s",
            (api_customer_id, limit),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Stripe -- prepaid credit purchases
# ---------------------------------------------------------------------------

def _get_stripe():
    if not settings.stripe_secret_key:
        raise BillingError(
            "Billing isn't configured on this server (STRIPE_SECRET_KEY is unset)."
        )
    import stripe
    stripe.api_key = settings.stripe_secret_key
    return stripe


def create_checkout_session(api_customer_id: str, pack_id: str) -> str:
    """Returns the Stripe-hosted Checkout URL to redirect the customer's
    browser to. credits/api_customer_id are stashed in the session's
    metadata (not inferred at webhook time) so fulfillment is a pure
    function of the webhook payload -- no risk of the pack's price or
    credit amount having changed between checkout and fulfillment."""
    pack = CREDIT_PACKS.get(pack_id)
    if pack is None:
        raise BillingError(
            f"Unknown credit pack '{pack_id}'. Available: {', '.join(CREDIT_PACKS)}"
        )

    stripe = _get_stripe()
    session = stripe.checkout.Session.create(
        mode="payment",
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": f"CodeSage API credits -- {pack['name']} ({pack['credits']:,} credits)",
                },
                "unit_amount": pack["price_cents"],
            },
            "quantity": 1,
        }],
        metadata={
            "api_customer_id": api_customer_id,
            "pack_id": pack_id,
            "credits": str(pack["credits"]),
        },
        success_url=settings.billing_success_url + "?session_id={CHECKOUT_SESSION_ID}",
        cancel_url=settings.billing_cancel_url,
    )
    return session.url


def fulfill_checkout_session(session: dict) -> None:
    """Called from the webhook handler once a checkout.session.completed
    event's signature has been verified (see app/main.py). Grants the
    credits from the session's own metadata, keyed on the session id so
    a Stripe redelivery never double-credits (see grant_credits'
    `reference` handling)."""
    metadata = session.get("metadata") or {}
    api_customer_id = metadata.get("api_customer_id")
    credits = metadata.get("credits")
    if not api_customer_id or not credits:
        raise BillingError("Checkout session is missing required metadata.")

    grant_credits(
        api_customer_id,
        amount=int(credits),
        reason=f"stripe_checkout:{metadata.get('pack_id', 'unknown')}",
        reference=session.get("id"),
    )


def construct_webhook_event(payload: bytes, signature_header: str | None) -> dict:
    """Verifies the Stripe-Signature header and returns the parsed
    event, or raises BillingError -- never trusts an unverified payload,
    since anyone could otherwise POST a fake checkout.session.completed
    and grant themselves free credits."""
    if not settings.stripe_webhook_secret:
        raise BillingError("STRIPE_WEBHOOK_SECRET is not configured on this server.")
    stripe = _get_stripe()
    try:
        return stripe.Webhook.construct_event(
            payload, signature_header, settings.stripe_webhook_secret
        )
    except (ValueError, Exception) as e:  # stripe.error.SignatureVerificationError subclasses Exception
        raise BillingError(f"Invalid Stripe webhook signature: {e}")
