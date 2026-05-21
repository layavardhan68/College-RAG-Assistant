"""
Fix SSL certificate errors on Windows / college networks.

Call configure_ssl() once at app startup before any Google API calls.
"""

import os
import ssl


def configure_ssl() -> None:
    """Point Python/httpx at certifi CA bundle (fixes many Windows SSL issues)."""
    try:
        import certifi

        ca_bundle = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
        os.environ.setdefault("CURL_CA_BUNDLE", ca_bundle)
    except ImportError:
        pass


def _ssl_verify_disabled() -> bool:
    return os.getenv("SSL_VERIFY", "true").lower() in ("0", "false", "no", "off")


def get_ssl_verify_setting():
    """
    Return httpx 'verify' for Google GenAI client_args.

    Must be an SSLContext (not False) — the Google SDK treats False as "unset".
    """
    configure_ssl()

    if _ssl_verify_disabled():
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()
