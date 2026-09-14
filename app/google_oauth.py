"""
Google OAuth ("Continue with Google") -- lets a user log in without a
password. Unlike app/github_oauth.py, this is authentication-only: no
Google access token is stored after login, since nothing here needs to
call a Google API on the user's behalf afterward (there's no Google
equivalent of "clone a private repo").

Flow mirrors github_oauth.py exactly -- see that module's docstring for
the general shape (login vs link intents, the shared callback pattern in
main.py). To use this, register an OAuth 2.0 Client ID at
https://console.cloud.google.com/apis/credentials with authorized redirect
URI {your backend url}/auth/google/callback, then set GOOGLE_CLIENT_ID and
GOOGLE_CLIENT_SECRET.
"""
from __future__ import annotations

import secrets
from urllib.parse import urlencode

import httpx

from app.config import settings

GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

GOOGLE_SCOPE = "openid email profile"


class GoogleOAuthError(Exception):
    """Raised for any failure in the OAuth exchange; main.py maps this to
    a 400 rather than letting a raw Google API error reach the client."""


def generate_state() -> str:
    return secrets.token_urlsafe(24)


def build_authorize_url(state: str) -> str:
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": GOOGLE_SCOPE,
        "state": state,
        # Always show the account chooser rather than silently reusing
        # whichever Google account happens to be active in the browser --
        # avoids someone on a shared machine accidentally signing into the
        # wrong Google identity's CodeSage account.
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code_for_token(code: str) -> str:
    resp = httpx.post(
        GOOGLE_TOKEN_URL,
        data={
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "code": code,
            "redirect_uri": settings.google_redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=10,
    )
    data = resp.json()
    if "error" in data:
        raise GoogleOAuthError(data.get("error_description", data["error"]))
    token = data.get("access_token")
    if not token:
        raise GoogleOAuthError("Google did not return an access token.")
    return token


def fetch_google_user(access_token: str) -> dict:
    """Returns {"id", "email", "email_verified"} for the token's owner.
    "id" is Google's stable `sub` claim -- a numeric-looking string, but
    NOT guaranteed to fit a 64-bit int the way GitHub's user id does
    (see db.py: stored as TEXT, not BIGINT, for exactly this reason).
    "email_verified" is Google's own claim about that address, trusted
    as-is -- see auth.find_or_create_google_user."""
    resp = httpx.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    if resp.status_code != 200:
        raise GoogleOAuthError("Couldn't fetch your Google profile.")
    data = resp.json()
    return {
        "id": data["sub"],
        "email": data.get("email"),
        "email_verified": bool(data.get("email_verified", False)),
    }
