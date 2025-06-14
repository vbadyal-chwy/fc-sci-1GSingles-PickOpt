#!/usr/bin/env python3
"""
Simple database validation script to check schema correctness.
"""

import sqlite3
import sys
from pathlib import Path

def validate_database(db_path: str):
    """Validate database structure and contents."""
    
    print(f"Validating database: {db_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys for this connection
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"✓ Tables ({len(tables)}): {', '.join(tables)}")
        
        # Check views  
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")
        views = [row[0] for row in cursor.fetchall()]
        print(f"✓ Views ({len(views)}): {', '.join(views)}")
        
        # Check indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        indexes = [row[0] for row in cursor.fetchall()]
        print(f"✓ Indexes ({len(indexes)}): {len(indexes)} custom indexes created")
        
        # Check schema version
        cursor.execute("SELECT version, description FROM schema_version")
        version_info = cursor.fetchone()
        print(f"✓ Schema version: {version_info[0]} - {version_info[1]}")
        
        # Check foreign key constraints (test with a simple query)
        cursor.execute("PRAGMA foreign_keys")
        fk_enabled = cursor.fetchone()[0]
        print(f"✓ Foreign key constraints: {'enabled' if fk_enabled else 'disabled'}")
        
        # Test a simple view query
        cursor.execute("SELECT COUNT(*) FROM v_execution_summary")
        count = cursor.fetchone()[0]
        print(f"✓ Views working: v_execution_summary has {count} rows")
        
        conn.close()
        print("\n✅ Database validation successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Database validation failed: {e}")
        return False

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "simulation.db"
    validate_database(db_path) 