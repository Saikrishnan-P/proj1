"""
Sends transactional email via plain SMTP -- works with Gmail (an app
password), SendGrid, Mailgun, AWS SES, or any other provider that exposes
an SMTP relay, using only Python's stdlib smtplib/email (no new dependency).

Best-effort by design: a failure here must never break signup itself (see
main.py's POST /auth/signup) -- it's logged and swallowed, not raised,
since "the account was created but the confirmation email bounced" is a
far better failure mode than "signup itself failed because an email
provider had a bad minute." A user who never got the email can always hit
POST /auth/resend-verification once the underlying SMTP issue is fixed.
"""
from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.config import settings


def send_verification_email(to_email: str, verify_url: str) -> bool:
    """Returns True if the email was handed off to the SMTP server
    successfully, False if sending failed for any reason (logged, not
    raised -- see module docstring)."""
    if not settings.smtp_host:
        # No SMTP configured -- expected in local dev. Print the link so
        # signup/verification is still testable end-to-end without real
        # email credentials, instead of silently doing nothing.
        print(f"[email] SMTP not configured -- verification link for {to_email}: {verify_url}")
        return False

    subject = "Verify your CodeSage email address"
    text_body = (
        "Welcome to CodeSage!\n\n"
        f"Confirm your email address by visiting:\n{verify_url}\n\n"
        f"This link expires in {settings.email_verification_ttl_hours} hours. "
        "If you didn't create a CodeSage account, you can safely ignore this email."
    )
    html_body = f"""\
<div style="font-family: sans-serif; max-width: 480px; margin: 0 auto;">
  <h2>Welcome to CodeSage</h2>
  <p>Confirm your email address to finish setting up your account.</p>
  <p>
    <a href="{verify_url}"
       style="display:inline-block; background:#3FB950; color:#0B0E14;
              padding:10px 20px; border-radius:4px; text-decoration:none;
              font-weight:bold;">
      Verify email address
    </a>
  </p>
  <p style="color:#8B95A7; font-size:13px;">
    This link expires in {settings.email_verification_ttl_hours} hours.
    If you didn't create a CodeSage account, you can safely ignore this email.
  </p>
</div>
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.smtp_from_email
    msg["To"] = to_email
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as server:
            if settings.smtp_use_tls:
                server.starttls()
            if settings.smtp_username:
                server.login(settings.smtp_username, settings.smtp_password)
            server.sendmail(settings.smtp_from_email, [to_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[email] Failed to send verification email to {to_email}: {e}")
        return False
