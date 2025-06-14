"""
Model trigger logic utilities for TF and TA execution decisions.

This module provides functions to determine when Tour Formation and Tour Allocation
models should be triggered based on various conditions including backlog levels,
time intervals, and tour inventory.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


def get_container_backlog_count(
    execution_id: str,
    db_path: Path,
    wh_id: str,
    planning_timestamp: datetime
) -> int:
    """
    Get current container backlog count (same logic as TF orchestrator).
    
    Args:
        execution_id: Database execution ID
        db_path: Path to SQLite database
        wh_id: Warehouse ID
        planning_timestamp: Current planning timestamp
        
    Returns:
        Number of unreleased containers in backlog
    """
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT COUNT(DISTINCT container_id) as backlog_count
            FROM containers
            WHERE execution_id = ?
            AND wh_id = ?
            AND arrive_datetime <= ?
            AND released_flag = 0
            """
            
            result = conn.execute(query, [
                execution_id, 
                wh_id, 
                planning_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            ]).fetchone()
            
            return result[0] if result else 0
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to get container backlog count: {e}")
        return 0


def get_ready_tours_count(
    execution_id: str,
    db_path: Path,
    wh_id: str
) -> int:
    """
    Get count of unarchived ready tours.
    
    Args:
        execution_id: Database execution ID
        db_path: Path to SQLite database
        wh_id: Warehouse ID
        
    Returns:
        Number of ready tours available for TA
    """
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT COUNT(DISTINCT tour_id) as ready_count
            FROM ready_to_release_tours
            WHERE execution_id = ?
            AND wh_id = ?
            AND archived_flag = 0
            """
            
            result = conn.execute(query, [execution_id, wh_id]).fetchone()
            return result[0] if result else 0
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to get ready tours count: {e}")
        return 0


def get_last_successful_tf_timestamp(
    execution_id: str,
    db_path: Path,
    wh_id: str
) -> Optional[datetime]:
    """
    Get timestamp of last successful TF run.
    
    Args:
        execution_id: Database execution ID
        db_path: Path to SQLite database
        wh_id: Warehouse ID
        
    Returns:
        Datetime of last successful TF run, or None if no previous runs
    """
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT MAX(planning_datetime) as last_tf_time
            FROM tf_clustering_metadata
            WHERE execution_id = ?
            AND wh_id = ?
            """
            
            result = conn.execute(query, [execution_id, wh_id]).fetchone()
            
            if result and result[0]:
                return datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
            return None
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to get last TF timestamp: {e}")
        return None


def check_tf_trigger_conditions(
    execution_id: str,
    db_path: Path,
    wh_id: str,
    current_time: datetime,
    config: dict,
    last_tf_time: Optional[datetime] = None,
    is_first_iteration: bool = False
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check all TF trigger conditions and return decision with metadata.
    
    Args:
        execution_id: Database execution ID
        db_path: Path to SQLite database
        wh_id: Warehouse ID
        current_time: Current simulation time
        config: Configuration dictionary
        last_tf_time: Last successful TF timestamp
        is_first_iteration: Whether this is the first simulation iteration
        
    Returns:
        Tuple of (should_trigger, reason, metadata_dict)
    """
    pick_opt_config = config.get('pick_optimization', {})
    
    # Get configuration values
    tf_min_containers = pick_opt_config.get('tf_min_containers_in_backlog', 1000)
    target_tours = pick_opt_config.get('target_tours_per_iteration', 5)
    tf_interval_minutes = pick_opt_config.get('tf_interval_minutes', 30)
    tf_demand_trigger_enabled = pick_opt_config.get('tf_demand_trigger_enabled', True)
    
    # Get current counts
    backlog_count = get_container_backlog_count(execution_id, db_path, wh_id, current_time)
    ready_tours_count = get_ready_tours_count(execution_id, db_path, wh_id)
    
    # Calculate time since last TF
    minutes_since_last_tf = None
    if last_tf_time:
        time_diff = current_time - last_tf_time
        minutes_since_last_tf = int(time_diff.total_seconds() / 60)
    
    # Check primary condition: sufficient backlog
    backlog_condition_met = backlog_count >= tf_min_containers
    
    # Prepare metadata
    metadata = {
        'backlog_container_count': backlog_count,
        'tf_min_containers_threshold': tf_min_containers,
        'backlog_condition_met': backlog_condition_met,
        'ready_tours_count': ready_tours_count,
        'target_tours_threshold': target_tours,
        'minutes_since_last_tf': minutes_since_last_tf,
        'is_first_iteration': is_first_iteration,
        'tf_demand_trigger_enabled': tf_demand_trigger_enabled
    }
    
    # If insufficient backlog, don't trigger
    if not backlog_condition_met:
        reason = f"Insufficient container backlog ({backlog_count} < {tf_min_containers})"
        return False, reason, metadata
    
    # With sufficient backlog, check additional trigger conditions
    trigger_reasons = []
    
    # First iteration trigger
    if is_first_iteration:
        trigger_reasons.append("first iteration")
    
    # Time-based trigger
    if minutes_since_last_tf is not None and minutes_since_last_tf >= tf_interval_minutes:
        trigger_reasons.append(f"time interval ({minutes_since_last_tf} >= {tf_interval_minutes} minutes)")
    elif last_tf_time is None:
        trigger_reasons.append("no previous TF run")
    
    # Demand-based trigger
    if tf_demand_trigger_enabled and ready_tours_count <= target_tours:
        trigger_reasons.append(f"low tour inventory ({ready_tours_count} <= {target_tours})")
    
    # Make decision
    if trigger_reasons:
        reason = f"Sufficient backlog with {' + '.join(trigger_reasons)}"
        return True, reason, metadata
    else:
        reason = f"Sufficient backlog but no additional triggers (ready tours: {ready_tours_count}, time since last: {minutes_since_last_tf}min)"
        return False, reason, metadata


def check_ta_trigger_conditions(
    execution_id: str,
    db_path: Path,
    wh_id: str,
    config: dict
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check TA trigger conditions.
    
    TA is evaluated every 5 minutes (time-based scheduling) but only triggers
    when sufficient ready tours are available.
    
    Args:
        execution_id: Database execution ID
        db_path: Path to SQLite database
        wh_id: Warehouse ID
        config: Configuration dictionary
        
    Returns:
        Tuple of (should_trigger, reason, metadata_dict)
        
    Note:
        This function is called every ta_interval_minutes (5 minutes) by the
        simulation loop. It only checks tour availability, not timing.
    """
    pick_opt_config = config.get('pick_optimization', {})
    target_tours = pick_opt_config.get('ta_interval_minutes', 5)
    
    # Get ready tours count
    ready_tours_count = get_ready_tours_count(execution_id, db_path, wh_id)
    
    # Prepare metadata
    metadata = {
        'ready_tours_count': ready_tours_count,
        'target_tours_threshold': target_tours,
        'ta_ready_tours_sufficient': ready_tours_count >= target_tours
    }
    
    # TA triggers when sufficient tours are available (time-based evaluation handled by simulation loop)
    if ready_tours_count >= target_tours:
        reason = f"Scheduled evaluation with sufficient ready tours ({ready_tours_count} >= {target_tours})"
        return True, reason, metadata
    else:
        reason = f"Scheduled evaluation but insufficient ready tours ({ready_tours_count} < {target_tours})"
        return False, reason, metadata


def record_trigger_decision(
    db_path: Path,
    execution_id: str,
    planning_datetime: datetime,
    model_type: str,
    triggered: bool,
    reason: str,
    metadata: Dict[str, Any]
) -> None:
    """
    Record trigger decision in database for analysis.
    
    Args:
        db_path: Path to SQLite database
        execution_id: Database execution ID
        planning_datetime: Current planning timestamp
        model_type: 'TF' or 'TA'
        triggered: Whether model was triggered
        reason: Reason for trigger decision
        metadata: Additional metadata from trigger check
    """
    try:
        with sqlite3.connect(db_path) as conn:
            # Create table if it doesn't exist
            conn.execute("""
            CREATE TABLE IF NOT EXISTS trigger_decisions (
                execution_id TEXT NOT NULL,
                planning_datetime TIMESTAMP NOT NULL,
                model_type TEXT NOT NULL,
                triggered BOOLEAN NOT NULL,
                trigger_reason TEXT,
                backlog_container_count INTEGER,
                tf_min_containers_threshold INTEGER,
                backlog_condition_met BOOLEAN,
                ready_tours_count INTEGER,
                target_tours_threshold INTEGER,
                minutes_since_last_tf INTEGER,
                ta_ready_tours_sufficient BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (execution_id, planning_datetime, model_type)
            )""")
            
            # Insert trigger decision record
            conn.execute("""
            INSERT OR REPLACE INTO trigger_decisions (
                execution_id, planning_datetime, model_type, triggered, trigger_reason,
                backlog_container_count, tf_min_containers_threshold, backlog_condition_met,
                ready_tours_count, target_tours_threshold, minutes_since_last_tf,
                ta_ready_tours_sufficient
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                execution_id,
                planning_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                model_type,
                triggered,
                reason,
                metadata.get('backlog_container_count'),
                metadata.get('tf_min_containers_threshold'),
                metadata.get('backlog_condition_met'),
                metadata.get('ready_tours_count'),
                metadata.get('target_tours_threshold'),
                metadata.get('minutes_since_last_tf'),
                metadata.get('ta_ready_tours_sufficient')
            ])
            
            conn.commit()
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to record trigger decision: {e}") 