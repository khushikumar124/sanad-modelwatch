"""Print the environment lines needed to enable login.

    python -m sanad.create_user alice

Prompts for a password (never echoed, never taken as an argument -- a
password on the command line lands in your shell history), then prints the
two variables to export. Nothing is written to disk: the credentials live
in your environment, so this file staying in the repository is harmless.

Prints are intentional here; this is an operator script, not library code.
"""
from __future__ import annotations

import getpass
import secrets
import sys

from sanad.api.auth import hash_password


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m sanad.create_user <username>", file=sys.stderr)
        return 2
    username = sys.argv[1].strip()
    if not username or ":" in username or "," in username:
        print("username must be non-empty and contain no ':' or ','", file=sys.stderr)
        return 2

    password = getpass.getpass("Password: ")
    if len(password) < 8:
        print("password must be at least 8 characters", file=sys.stderr)
        return 2
    if password != getpass.getpass("Confirm password: "):
        print("passwords did not match", file=sys.stderr)
        return 2

    print("\nAdd these to your shell before starting Sanad:\n")
    print("export SANAD_AUTH_ENABLED=true")
    print(f"export SANAD_SESSION_SECRET={secrets.token_hex(32)}")
    print(f'export SANAD_USERS="{username}:{hash_password(password)}"')
    print(
        "\nKeep the secret stable -- changing it invalidates every existing session.\n"
        "For more than one user, comma-separate the entries in SANAD_USERS."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
