"""Sanitize exported records and service errors without retaining credentials."""
import re

SECRET_KEYS = {"api_key", "deepseek_api_key", "authorization", "access_token", "secret", "password"}


def redact(value, secrets=()):
    if isinstance(value, dict):
        return {str(k): redact(v, secrets) for k, v in value.items()
                if str(k).lower() not in SECRET_KEYS}
    if isinstance(value, (list, tuple)):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", value)
    return value
