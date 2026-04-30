from __future__ import annotations

import argparse
import hashlib
import sqlite3
from pathlib import Path

DEFAULT_KEEP = [
    "dr.arvind.rao@telehealth-vision.in",
    "dr.sneha.desai@telehealth-vision.in",
    "dr.vikram.singh@telehealth-vision.in",
    "dr.priya.mehta@telehealth-vision.in",
]


def build_password_hash(password: str, salt: str = "dev-salt-change-me", iters: int = 210_000) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iters)
    return f"pbkdf2_sha256${iters}${salt}${dk.hex()}"


def reset_sqlite(db_file: Path, keep_emails: list[str], password: str) -> dict[str, object]:
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.execute("DELETE FROM scans")
    cur.execute("DELETE FROM patients")
    cur.execute("DELETE FROM doctors")

    password_hash = build_password_hash(password)
    for email in keep_emails:
        cur.execute(
            "INSERT INTO doctors (email, password_hash, created_at) VALUES (?, ?, datetime('now'))",
            (email, password_hash),
        )

    conn.commit()
    cur.execute("SELECT email FROM doctors ORDER BY email")
    rows = [row[0] for row in cur.fetchall()]
    conn.close()

    return {
        "doctor_count": len(rows),
        "doctors": rows,
        "patient_count": 0,
        "scan_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset local SQLite tables and keep only fixed doctor accounts.")
    parser.add_argument(
        "--db",
        default="backend/ophthalmic_ai.db",
        help="Path to SQLite DB file (default: backend/ophthalmic_ai.db)",
    )
    parser.add_argument(
        "--password",
        default="password123",
        help="Password to set for all kept doctors (default: password123)",
    )
    parser.add_argument(
        "--keep",
        nargs="*",
        default=DEFAULT_KEEP,
        help="Doctor emails to keep (default: 4 configured doctors)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    keep = [item.strip().lower() for item in args.keep if item.strip()]
    if not keep:
        raise ValueError("At least one doctor email is required in --keep")

    result = reset_sqlite(db_path, keep, args.password)
    print(result)


if __name__ == "__main__":
    main()
