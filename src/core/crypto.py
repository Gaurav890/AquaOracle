"""Encryption at rest for stored provider API keys.

OpenAI/Anthropic keys are stored in the local database (per the user's
explicit request: "everything stored in local database... nothing goes to
[a third party]"), but plaintext-at-rest for live API credentials is bad
practice regardless of where the database lives. Fernet (symmetric,
authenticated) keyed by Settings.secret_key is enough for this threat model
— it's not protecting against someone with direct filesystem access to the
whole app anyway, just against casual exposure (backups, accidental commits
of the DB file, etc.).
"""

from cryptography.fernet import Fernet, InvalidToken

from src.core.config import settings


def _get_fernet() -> Fernet:
    if not settings.secret_key:
        raise RuntimeError(
            "SECRET_KEY is not set. Generate one with:\n"
            "  python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"\n"
            "and add it to .env as SECRET_KEY=..."
        )
    return Fernet(settings.secret_key.encode())


def encrypt_api_key(plaintext: str) -> bytes:
    return _get_fernet().encrypt(plaintext.encode())


def decrypt_api_key(ciphertext: bytes) -> str:
    try:
        return _get_fernet().decrypt(ciphertext).decode()
    except InvalidToken:
        raise ValueError("Stored API key could not be decrypted (wrong SECRET_KEY?)")
