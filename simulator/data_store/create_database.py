#!/usr/bin/env python3
"""
Database creation script for Pick Optimization.

This script creates the SQLite database using the schema definition.
"""

import sqlite3
import logging
from pathlib import Path


def create_database(db_path: str, schema_path: str = None) -> bool:
    """
    Create the Pick Optimization SQLite database.
    
    Parameters
    ----------
    db_path : str
        Path where the database file should be created
    schema_path : str, optional
        Path to the SQL schema file, defaults to database_schema.sql in same directory
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Set default schema path
    if schema_path is None:
        schema_path = Path(__file__).parent / "database_schema.sql"
    
    logger.info(f"Creating database: {db_path}")
    logger.info(f"Using schema: {schema_path}")
    
    try:
        # Read schema file
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Create database and execute schema
        with sqlite3.connect(db_path) as conn:
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Execute schema (split by semicolon to handle multiple statements)
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            
            for statement in statements:
                if statement:
                    try:
                        conn.execute(statement)
                    except sqlite3.Error as e:
                        logger.warning(f"Non-critical SQL warning: {e}")
                        # Continue with remaining statements
            
            conn.commit()
        
        logger.info(f"Database created successfully: {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        return False


def main():
    """Command line interface for database creation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Pick Optimization database")
    parser.add_argument('--database', '-d', required=True, help="Database file path")
    parser.add_argument('--schema', '-s', help="Schema file path (optional)")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create database
    success = create_database(args.database, args.schema)
    
    if success:
        print(f"Database created successfully: {args.database}")
        exit(0)
    else:
        print(f"Failed to create database: {args.database}")
        exit(1)


if __name__ == "__main__":
    main() 