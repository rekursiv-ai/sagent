"""The tool server's certificate, and the TLS a client pins it with.

The server signs its own certificate; no authority vouches for it. A
connection string carries its fingerprint instead, and a client checks the
fingerprint before it sends the token, as ssh checks a host key.

Typical usage example:

  pinned = fingerprint(directory)            # Made on first use.
  server = await asyncio.start_server(..., ssl=server_context(directory))
  reader, writer = await asyncio.open_connection(host, port, ssl=client_context())
  if peer_fingerprint(writer) != pinned: ...
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

import hashlib
import socket
import ssl

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from sagent.lib.files.owner_only import write_private


if TYPE_CHECKING:
    from pathlib import Path

    import asyncio


__all__ = [
    "client_context",
    "fingerprint",
    "peer_fingerprint",
    "server_context",
]


def fingerprint(directory: Path) -> str:
    """Return the fingerprint of the certificate in ``directory``, made if there is none.

    Args:
      directory: Where ``key.pem`` and ``cert.pem`` are kept, readable by their
        owner alone.

    Returns:
      fingerprint: ``SHA256:`` and the hex digest of the certificate's DER bytes.

    """
    cert = directory / "cert.pem"
    if not cert.exists():
        _make(directory)
    return _der_fingerprint(ssl.PEM_cert_to_DER_cert(cert.read_text()))


def server_context(directory: Path) -> ssl.SSLContext:
    """Return the server's TLS context, its certificate made if there is none."""
    _ = fingerprint(directory)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(directory / "cert.pem", directory / "key.pem")
    return context


# No authority signed the certificate, so none is checked; ``peer_fingerprint``
# against the connection string's pin is the check, made before the token is sent.
def client_context() -> ssl.SSLContext:
    """Return a client's TLS context, which leaves the server's identity to the pin."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def peer_fingerprint(writer: asyncio.StreamWriter) -> str:
    """Return the fingerprint of the certificate the server of ``writer`` showed.

    Args:
      writer: A connection opened with ``client_context``.

    Returns:
      fingerprint: ``SHA256:`` and the hex digest of the certificate's DER bytes.

    Raises:
      ValueError: The connection is no TLS, or the server showed no certificate.

    """
    ssl_object = cast(object, writer.get_extra_info("ssl_object"))
    der = (
        ssl_object.getpeercert(binary_form=True)
        if isinstance(ssl_object, ssl.SSLObject)
        else None
    )
    if der is None:
        raise ValueError("The server showed no certificate.")
    return _der_fingerprint(der)


def _der_fingerprint(der: bytes) -> str:
    return f"SHA256:{hashlib.sha256(der).hexdigest()}"


def _make(directory: Path) -> None:
    """Write a new key and a certificate it signs, for ten years, to ``directory``."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, socket.gethostname())])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )
    write_private(
        directory / "key.pem",
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )
    write_private(directory / "cert.pem", cert.public_bytes(serialization.Encoding.PEM))
