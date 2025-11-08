#!/usr/bin/env python
"""
Change Airflow Admin Password
Updates the admin user password in Airflow database using SQLite directly.

Usage:
    python scripts/change_airflow_password.py
"""

import sqlite3
from pathlib import Path


def change_password():
    """Change admin user password in Airflow database"""

    airflow_home = Path.home() / 'portfolio-insight-agent' / 'airflow'
    db_path = airflow_home / 'airflow.db'
    new_password = "admin123"

    if not db_path.exists():
        print(f"✗ Database not found at {db_path}")
        return 1

    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Update password for admin user
        cursor.execute("""
            UPDATE ab_user
            SET password = ?
            WHERE username = 'admin'
        """, (new_password,))

        if cursor.rowcount > 0:
            conn.commit()
            print(f"✓ Password changed successfully for user 'admin'")
            print(f"  New password: {new_password}")
            print(f"\n  Login at: http://localhost:8080")
            print(f"  Username: admin")
            print(f"  Password: {new_password}")
        else:
            print("✗ Admin user not found in database")
            return 1

        conn.close()
        return 0

    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(change_password())
