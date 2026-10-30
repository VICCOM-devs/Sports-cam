"""Self-signed certificate for the HTTPS listener.

Phones only allow camera access on a secure origin, so camera phones open the hub over HTTPS
and accept the hub's certificate once. Viewers use plain HTTP (WebRTC playback does not need
a secure origin). The certificate covers the hub's LAN IP and 'localhost'.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


def ensure_cert(certs_dir: Path, host_ip: str) -> tuple[Path, Path]:
    cert_file, key_file = certs_dir / "hub.crt", certs_dir / "hub.key"
    if cert_file.exists() and key_file.exists():
        try:
            cert = x509.load_pem_x509_certificate(cert_file.read_bytes())
            sans = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            ips = {str(ip) for ip in sans.get_values_for_type(x509.IPAddress)}
            if host_ip in ips and cert.not_valid_after_utc > dt.datetime.now(dt.timezone.utc):
                return cert_file, key_file
        except Exception:
            pass  # regenerate

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AnySport hub")])
    san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        san.append(x509.IPAddress(ipaddress.ip_address(host_ip)))
    except ValueError:
        san.append(x509.DNSName(host_ip))
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    certs_dir.mkdir(parents=True, exist_ok=True)
    key_file.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_file, key_file
