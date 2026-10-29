"""Login rate limiting keys on the client, not on a trusted reverse proxy.

Behind a reverse proxy every request's socket peer is the proxy. The password
and passkey login limiters must key on the forwarded client IP when
HERMES_WEBUI_TRUST_FORWARDED_FOR=1 and the raw peer is listed in
HERMES_WEBUI_TRUSTED_PROXY_CIDRS, and on the raw socket peer otherwise, so a
direct client cannot pick a fresh bucket by sending forwarded headers.
"""
import http.client
import io
from types import SimpleNamespace

import pytest

from api import auth, passkeys, routes

LOGIN_PATHS = ["/api/auth/login", "/api/auth/passkey/login"]


class _Handler:
    def __init__(self, peer, headers):
        self.client_address = (peer, 12345)
        self.headers = http.client.HTTPMessage()
        self.headers["Content-Length"] = "0"
        # A list of pairs sends repeated header fields.
        for key, value in headers.items() if isinstance(headers, dict) else headers:
            self.headers[key] = value
        self.rfile = io.BytesIO(b"")
        self.wfile = io.BytesIO()
        self.status = None

    def send_response(self, code):
        self.status = code

    def send_header(self, key, value):
        pass

    def end_headers(self):
        pass


@pytest.fixture(autouse=True)
def _failing_logins(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "_LOGIN_ATTEMPTS_FILE", tmp_path / ".login_attempts.json")
    monkeypatch.setattr(auth, "_login_attempts", {})
    monkeypatch.setattr(auth, "is_auth_enabled", lambda: True)
    monkeypatch.setattr(auth, "_passkey_feature_flag_enabled", lambda: True)
    monkeypatch.setattr(auth, "verify_password", lambda password: False)

    def reject(body, handler):
        raise passkeys.PasskeyError("assertion rejected")

    monkeypatch.setattr(passkeys, "finish_login", reject)


def _login(path, peer, headers):
    handler = _Handler(peer, headers)
    routes.handle_post(handler, SimpleNamespace(path=path))
    return handler.status


@pytest.mark.parametrize("path", LOGIN_PATHS)
def test_failed_logins_behind_trusted_proxy_do_not_lock_out_other_clients(path, monkeypatch):
    monkeypatch.setenv("HERMES_WEBUI_TRUST_FORWARDED_FOR", "1")
    monkeypatch.setenv("HERMES_WEBUI_TRUSTED_PROXY_CIDRS", "127.0.0.1/32")
    attacker = {"X-Forwarded-For": "203.0.113.10"}
    for _ in range(auth._LOGIN_MAX_ATTEMPTS):
        assert _login(path, "127.0.0.1", attacker) == 401

    assert _login(path, "127.0.0.1", attacker) == 429
    assert _login(path, "127.0.0.1", {"X-Forwarded-For": "203.0.113.20"}) == 401


@pytest.mark.parametrize("path", LOGIN_PATHS)
@pytest.mark.parametrize(
    "trust_forwarded, peer, header",
    [
        # Opt-in off: forwarded headers are ignored even from loopback.
        (False, "127.0.0.1", "X-Forwarded-For"),
        # Opt-in on, but the peer is not a trusted proxy.
        (True, "198.51.100.7", "X-Forwarded-For"),
        # Loopback is not enough: a direct or SSH-tunnel client is loopback too.
        (True, "127.0.0.1", "X-Forwarded-For"),
        # Listed proxy, but X-Real-IP is not an IP address.
        (True, "10.0.0.5", "X-Real-IP"),
    ],
)
def test_unusable_forwarded_value_keys_on_socket_peer(path, trust_forwarded, peer, header, monkeypatch):
    monkeypatch.setenv("HERMES_WEBUI_TRUST_FORWARDED_FOR", "1" if trust_forwarded else "")
    monkeypatch.setenv("HERMES_WEBUI_TRUSTED_PROXY_CIDRS", "10.0.0.5/32")
    # A fresh forwarded value on every attempt must not buy a fresh bucket.
    values = [
        f"client-{i}" if header == "X-Real-IP" else f"203.0.113.{i}"
        for i in range(auth._LOGIN_MAX_ATTEMPTS + 1)
    ]
    for value in values[:-1]:
        assert _login(path, peer, {header: value}) == 401

    assert _login(path, peer, {header: values[-1]}) == 429


@pytest.mark.parametrize("path", LOGIN_PATHS)
@pytest.mark.parametrize("real_ip", ["198.51.100.9", "not-an-ip"])
def test_rotating_xff_with_fixed_real_ip_keys_on_socket_peer(path, real_ip, monkeypatch):
    # The proxy sets X-Real-IP but passes the client's own X-Forwarded-For through.
    monkeypatch.setenv("HERMES_WEBUI_TRUST_FORWARDED_FOR", "1")
    monkeypatch.setenv("HERMES_WEBUI_TRUSTED_PROXY_CIDRS", "10.0.0.5/32")
    for i in range(auth._LOGIN_MAX_ATTEMPTS):
        headers = {"X-Forwarded-For": f"203.0.113.{i}", "X-Real-IP": real_ip}
        assert _login(path, "10.0.0.5", headers) == 401

    headers = {"X-Forwarded-For": "203.0.113.99", "X-Real-IP": real_ip}
    assert _login(path, "10.0.0.5", headers) == 429


@pytest.mark.parametrize("path", LOGIN_PATHS)
@pytest.mark.parametrize(
    "fields",
    [
        # Empty first X-Forwarded-For field, rotating second one.
        lambda i: [("X-Forwarded-For", ""), ("X-Forwarded-For", f"203.0.113.{i}"), ("X-Real-IP", "198.51.100.9")],
        # Repeated X-Real-IP fields that disagree.
        lambda i: [("X-Forwarded-For", f"203.0.113.{i}"), ("X-Real-IP", f"203.0.113.{i}"), ("X-Real-IP", "198.51.100.9")],
        lambda i: [("X-Real-IP", f"203.0.113.{i}"), ("X-Real-IP", "198.51.100.9")],
    ],
)
def test_repeated_forwarded_fields_key_on_socket_peer(path, fields, monkeypatch):
    # The agreement check must see every field the resolver reads.
    monkeypatch.setenv("HERMES_WEBUI_TRUST_FORWARDED_FOR", "1")
    monkeypatch.setenv("HERMES_WEBUI_TRUSTED_PROXY_CIDRS", "10.0.0.5/32")
    for i in range(auth._LOGIN_MAX_ATTEMPTS):
        assert _login(path, "10.0.0.5", fields(i)) == 401

    assert _login(path, "10.0.0.5", fields(99)) == 429


@pytest.mark.parametrize("path", LOGIN_PATHS)
@pytest.mark.parametrize(
    "forwarded, real_ip",
    [
        ("203.0.113.10", "203.0.113.10"),
        ("2001:DB8::1", "2001:db8:0:0:0:0:0:1"),
        # An appending proxy keeps whatever the client sent to the left.
        ("1.1.1.1, 203.0.113.10", "203.0.113.10"),
    ],
)
def test_xff_agreeing_with_real_ip_keeps_per_client_buckets(path, forwarded, real_ip, monkeypatch):
    monkeypatch.setenv("HERMES_WEBUI_TRUST_FORWARDED_FOR", "1")
    monkeypatch.setenv("HERMES_WEBUI_TRUSTED_PROXY_CIDRS", "10.0.0.5/32")
    client = {"X-Forwarded-For": forwarded, "X-Real-IP": real_ip}
    for _ in range(auth._LOGIN_MAX_ATTEMPTS):
        assert _login(path, "10.0.0.5", client) == 401

    assert _login(path, "10.0.0.5", client) == 429
    other = {"X-Forwarded-For": "203.0.113.20", "X-Real-IP": "203.0.113.20"}
    assert _login(path, "10.0.0.5", other) == 401


@pytest.mark.parametrize("path", LOGIN_PATHS)
@pytest.mark.parametrize(
    "fields",
    [
        # A proxy that forwards no X-Forwarded-For but relays the client's own
        # X-Real-IP (nginx's default without proxy_set_header X-Real-IP).
        lambda i: [("X-Real-IP", f"203.0.113.{i}")],
        lambda i: [("X-Real-IP", f"2001:db8::{i + 1:x}")],
        # A blank X-Forwarded-For field is not a chain either.
        lambda i: [("X-Forwarded-For", " "), ("X-Real-IP", f"203.0.113.{i}")],
    ],
)
def test_real_ip_without_xff_keys_on_socket_peer(path, fields, monkeypatch):
    """Exp-v0.52.546..571 regression: a valid, rotating X-Real-IP with no
    X-Forwarded-For bought a fresh bucket per attempt (unlimited guesses) where
    v0.52.544 keyed on the socket peer and returned 429 from attempt 6."""
    monkeypatch.setenv("HERMES_WEBUI_TRUST_FORWARDED_FOR", "1")
    monkeypatch.setenv("HERMES_WEBUI_TRUSTED_PROXY_CIDRS", "127.0.0.1/32")
    for i in range(auth._LOGIN_MAX_ATTEMPTS):
        assert _login(path, "127.0.0.1", fields(i)) == 401

    assert _login(path, "127.0.0.1", fields(99)) == 429


@pytest.mark.parametrize("path", LOGIN_PATHS)
def test_xff_without_real_ip_keeps_per_client_buckets(path, monkeypatch):
    """The fix must not cost the XFF-only setup its per-client buckets."""
    monkeypatch.setenv("HERMES_WEBUI_TRUST_FORWARDED_FOR", "1")
    monkeypatch.setenv("HERMES_WEBUI_TRUSTED_PROXY_CIDRS", "127.0.0.1/32")
    client = {"X-Forwarded-For": "1.1.1.1, 203.0.113.10"}
    for _ in range(auth._LOGIN_MAX_ATTEMPTS):
        assert _login(path, "127.0.0.1", client) == 401

    assert _login(path, "127.0.0.1", client) == 429
    assert _login(path, "127.0.0.1", {"X-Forwarded-For": "203.0.113.20"}) == 401
