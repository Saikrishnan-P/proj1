"""
GitHub OAuth ("Continue with GitHub") -- lets a user log in without a
password, and grants CodeSage an access token scoped to read their repos
(including private ones) for cloning during ingestion.

Flow:
    1. Frontend links/redirects to GET /auth/github/login (see main.py),
       which redirects the browser on to GitHub's own consent screen.
    2. User approves; GitHub redirects back to
       GET /auth/github/callback?code=...&state=...
    3. main.py exchanges that code for an access token (server-to-server,
       via this module), finds-or-creates a CodeSage account tied to the
       GitHub user id (see auth.find_or_create_github_user), stores the
       token encrypted, and issues a normal CodeSage JWT exactly like a
       password login would.
    4. main.py redirects the browser back to the frontend with that JWT.

To use this, register a GitHub OAuth App at
https://github.com/settings/developers with callback URL
{your backend url}/auth/github/callback, then set GITHUB_CLIENT_ID and
GITHUB_CLIENT_SECRET.
"""
from __future__ import annotations

import secrets
from urllib.parse import urlencode

import httpx

from app.config import settings

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"

# `repo` is what actually makes private-repo cloning work later -- public
# repos would clone fine with no scope at all, but there'd be no way to
# support private ones down the line without asking every existing user to
# re-authorize. read:user + user:email are just for the account/email
# lookup used to create the CodeSage account itself.
GITHUB_SCOPE = "read:user user:email repo"


class GitHubOAuthError(Exception):
    """Raised for any failure in the OAuth exchange; main.py maps this to
    a 400 rather than letting a raw GitHub API error reach the client."""


def generate_state() -> str:
    """A random, unguessable value threaded through the whole redirect
    round-trip and checked on the way back (see main.py) -- standard
    OAuth CSRF protection, so a malicious site can't trick a logged-in
    browser into completing an attacker-initiated GitHub login."""
    return secrets.token_urlsafe(24)


def build_authorize_url(state: str) -> str:
    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": settings.github_redirect_uri,
        "scope": GITHUB_SCOPE,
        "state": state,
        "allow_signup": "true",
    }
    return f"{GITHUB_AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code_for_token(code: str) -> str:
    resp = httpx.post(
        GITHUB_TOKEN_URL,
        headers={"Accept": "application/json"},
        data={
            "client_id": settings.github_client_id,
            "client_secret": settings.github_client_secret,
            "code": code,
            "redirect_uri": settings.github_redirect_uri,
        },
        timeout=10,
    )
    data = resp.json()
    if "error" in data:
        raise GitHubOAuthError(data.get("error_description", data["error"]))
    token = data.get("access_token")
    if not token:
        raise GitHubOAuthError("GitHub did not return an access token.")
    return token


def fetch_github_user(access_token: str) -> dict:
    """Returns {"id", "login", "email"} for the token's owner. `email` can
    be None even with user:email granted -- GitHub's /user endpoint omits
    it if the user's primary email is set to private, so the dedicated
    /user/emails endpoint (which respects the scope regardless of that
    visibility setting) is tried as a fallback."""
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github+json"}

    user_resp = httpx.get(GITHUB_USER_URL, headers=headers, timeout=10)
    if user_resp.status_code != 200:
        raise GitHubOAuthError("Couldn't fetch your GitHub profile.")
    user = user_resp.json()

    email = user.get("email")
    if not email:
        emails_resp = httpx.get(GITHUB_EMAILS_URL, headers=headers, timeout=10)
        if emails_resp.status_code == 200:
            primary = next((e for e in emails_resp.json() if e.get("primary")), None)
            email = primary["email"] if primary else None

    return {"id": user["id"], "login": user["login"], "email": email}
