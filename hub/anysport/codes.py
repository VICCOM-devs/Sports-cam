"""Match codes and role tokens.

Match code: 6 characters from an alphabet without look-alikes (no 0/O, 1/I/L), easy to
read off a printed QR sheet and type on a phone. Role tokens are random URL-safe strings.
"""
import secrets

ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def match_code(length: int = 6) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def normalize_code(raw: str) -> str:
    return (raw or "").strip().upper().replace(" ", "").replace("-", "")[:8]


def token(nbytes: int = 18) -> str:
    return secrets.token_urlsafe(nbytes)
