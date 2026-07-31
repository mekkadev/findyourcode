import hashlib
import hmac
import os
import time

SESSION_TTL = 3600


class CredentialChecker:
    """Validates the login/password pair a client presents at sign-in."""

    def __init__(self, users, secret: bytes):
        self.users = users
        self.secret = secret

    def check(self, login: str, password: str) -> str:
        record = self.users.get(login)
        if record is None:
            raise PermissionError("unknown login")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), record.salt, 200_000)
        if not hmac.compare_digest(digest, record.digest):
            raise PermissionError("wrong password")
        return self.issue_ticket(login)

    def issue_ticket(self, login: str) -> str:
        payload = f"{login}.{int(time.time()) + SESSION_TTL}".encode()
        mac = hmac.new(self.secret, payload, hashlib.sha256).hexdigest()
        return f"{payload.decode()}.{mac}"


def rotate_secret() -> bytes:
    return os.urandom(32)
