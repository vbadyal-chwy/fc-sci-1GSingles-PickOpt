-- Pick Optimization Database Schema
-- SQLite database for managing pick optimization data

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Schema versioning table
CREATE TABLE IF NOT EXISTS schema_version (
    version TEXT PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Insert current schema version
INSERT OR REPLACE INTO schema_version (version, description) 
VALUES ('2.0.0', 'Enhanced schema with TF/TA output tables and simulation support');

-- Executions table - tracks each simulation run
CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY,
    wh_id TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    tf_interval_minutes INTEGER NOT NULL,
    ta_interval_minutes INTEGER NOT NULL,
    tf_min_containers_in_backlog INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'INITIALIZED', -- INITIALIZED, RUNNING, COMPLETED, FAILED
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT -- JSON field for flexible configuration storage
);

-- Containers table - master container data
CREATE TABLE IF NOT EXISTS containers (
    execution_id TEXT NOT NULL,
    container_id TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    priority INTEGER DEFAULT 0,
    che_route TEXT,
    arrive_datetime DATETIME,
    original_promised_pull_datetime DATETIME,
    released_flag BOOLEAN DEFAULT FALSE,
    tour_id TEXT,
    release_datetime DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, container_id),
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);

-- Container details table - pick-level details for each container
CREATE TABLE IF NOT EXISTS container_details (
    execution_id TEXT NOT NULL,
    container_id TEXT NOT NULL,
    pick_id TEXT NOT NULL,
    item_number TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    planned_quantity REAL,
    pick_location TEXT,
    location_status TEXT,
    aisle_sequence INTEGER,
    aisle_name TEXT,
    picking_flow_as_int INTEGER,
    unit_volume REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, container_id, pick_id),
    FOREIGN KEY (execution_id, container_id) REFERENCES containers(execution_id, container_id) ON DELETE CASCADE
);

-- Slotbook table - inventory/location data
CREATE TABLE IF NOT EXISTS slotbook (
    execution_id TEXT NOT NULL,
    inventory_snapshot_date DATE NOT NULL,
    item_number TEXT NOT NULL,
    location_id TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    aisle_sequence INTEGER,
    picking_flow_as_int INTEGER,
    actual_qty REAL,
    type TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, inventory_snapshot_date, item_number, location_id),
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);

-- Labor data table - historical labor information from Vertica
CREATE TABLE IF NOT EXISTS labor_data (
    execution_id TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    date DATE NOT NULL,
    hour INTEGER NOT NULL,
    minutes INTEGER NOT NULL,
    count_employees INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, wh_id, date, hour, minutes),
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);


-- ============================================================================
-- Additional Output Tables for Enhanced Simulation Support
-- Added in Schema Version 2.0.0
-- ============================================================================

-- Container target calculations (TF input processing)
CREATE TABLE IF NOT EXISTS tf_container_target (
    execution_id TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    planning_datetime TIMESTAMP NOT NULL,
    active_headcount_multis INTEGER,
    historical_uph_multis REAL,
    avg_upc_multis REAL,
    container_target_variability_factor REAL,
    target_containers_per_interval INTEGER,
    backlog_container_count INTEGER,
    trigger_tf_flag BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, planning_datetime),
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);

-- Container slack calculations (TF preprocessing)
CREATE TABLE IF NOT EXISTS tf_container_slack (
    execution_id TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    planning_datetime TIMESTAMP NOT NULL,
    container_id TEXT NOT NULL,
    time_until_pull REAL,
    virtual_waiting_time REAL,
    picking_time REAL,
    travel_time REAL,
    pack_time REAL,
    break_impact REAL,
    total_processing_time REAL,
    slack_minutes REAL,
    slack_category TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, planning_datetime, container_id),
    FOREIGN KEY (execution_id, container_id) REFERENCES containers(execution_id, container_id) ON DELETE CASCADE
);

-- Container clustering assignments (TF output)
CREATE TABLE IF NOT EXISTS tf_container_clustering (
    execution_id TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    planning_datetime TIMESTAMP NOT NULL,
    container_id TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, planning_datetime, container_id),
    FOREIGN KEY (execution_id, container_id) REFERENCES containers(execution_id, container_id) ON DELETE CASCADE
);

-- Cluster metadata and performance metrics (TF output)
CREATE TABLE IF NOT EXISTS tf_clustering_metadata (
    execution_id TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    planning_datetime TIMESTAMP NOT NULL,
    cluster_id INTEGER NOT NULL,
    total_containers INTEGER,
    critical_containers INTEGER,
    planned_tour_count INTEGER,
    min_aisle INTEGER,
    max_aisle INTEGER,
    avg_centroid REAL,
    avg_span REAL,
    cluster_quality_score REAL,
    tour_id_offset INTEGER,
    solve_time REAL,
    objective_value REAL,
    total_distance REAL,
    total_slack REAL,
    actual_tour_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, planning_datetime, cluster_id),
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);

-- Generated tours with assignments (TF output)
CREATE TABLE IF NOT EXISTS tf_tour_formation (
    execution_id TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    planning_datetime TIMESTAMP NOT NULL,
    cluster_id INTEGER NOT NULL,
    tour_id TEXT NOT NULL,
    container_id TEXT NOT NULL,
    item_number TEXT NOT NULL,
    sku_volume REAL,
    pick_location TEXT,
    sequence_order INTEGER,
    pick_quantity INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, planning_datetime, tour_id, container_id, item_number, sequence_order),
    FOREIGN KEY (execution_id, container_id) REFERENCES containers(execution_id, container_id) ON DELETE CASCADE
);

-- Ready-to-release tour pool (persistent across TA iterations)
CREATE TABLE IF NOT EXISTS ready_to_release_tours (
    execution_id TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    tour_id TEXT NOT NULL,
    container_id TEXT NOT NULL,
    created_at_datetime TIMESTAMP NOT NULL, -- TF planning_datetime when created
    archived_flag BOOLEAN DEFAULT FALSE,
    archived_at_datetime TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, tour_id, container_id),
    FOREIGN KEY (execution_id, container_id) REFERENCES containers(execution_id, container_id) ON DELETE CASCADE
);

-- Pending tours by aisle (TA input)
CREATE TABLE IF NOT EXISTS ta_pending_tours_by_aisle (
    execution_id TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    planning_datetime TIMESTAMP NOT NULL,
    aisle INTEGER NOT NULL,
    tour_count INTEGER,
    quantity INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, planning_datetime, aisle),
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);

-- Tours selected for release by TA (TA output - based on tours_to_release.csv structure)
CREATE TABLE IF NOT EXISTS ta_tours_to_release (
    execution_id TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    planning_datetime TIMESTAMP NOT NULL,
    tour_id TEXT NOT NULL,
    container_id TEXT NOT NULL,
    item_number TEXT NOT NULL,  -- from sku column
    pick_qty INTEGER,           -- from quantity column
    location_id TEXT,           -- from optimal_pick_location column
    picking_flow_as_int INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, planning_datetime, tour_id, container_id, item_number),
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE,
    FOREIGN KEY (execution_id, container_id) REFERENCES containers(execution_id, container_id) ON DELETE CASCADE
);

-- Tour allocation metadata (TA output)
CREATE TABLE IF NOT EXISTS ta_allocation_metadata (
    execution_id TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    planning_datetime TIMESTAMP NOT NULL,
    solve_time REAL,
    tour_count_target INTEGER,
    tour_count_released INTEGER,
    total_aisle_concurrency INTEGER,
    maximum_aisle_concurrency INTEGER,
    total_slack REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, planning_datetime),
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);

-- Flexsim inputs (final output)
CREATE TABLE IF NOT EXISTS flexsim_inputs (
    execution_id TEXT NOT NULL,
    wh_id TEXT NOT NULL,
    planning_datetime TIMESTAMP NOT NULL,
    tour_id TEXT NOT NULL,
    container_id TEXT NOT NULL,
    item_number TEXT NOT NULL,
    pick_qty INTEGER,
    location_id TEXT,
    picking_flow_as_int INTEGER,
    aisle_sequence INTEGER,
    original_promised_pull_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (execution_id, planning_datetime, tour_id, container_id, item_number),
    FOREIGN KEY (execution_id, container_id) REFERENCES containers(execution_id, container_id) ON DELETE CASCADE
);

-- Planning iterations tracking table
CREATE TABLE IF NOT EXISTS planning_iterations (
    iteration_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    planning_datetime TIMESTAMP NOT NULL,
    iteration_type TEXT NOT NULL, -- 'TF', 'TA', or 'BOTH'
    status TEXT NOT NULL DEFAULT 'PENDING', -- PENDING, RUNNING, COMPLETED, FAILED
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_containers_wh_id ON containers(wh_id);
CREATE INDEX IF NOT EXISTS idx_containers_priority ON containers(priority);
CREATE INDEX IF NOT EXISTS idx_containers_released_flag ON containers(released_flag);
CREATE INDEX IF NOT EXISTS idx_containers_tour_id ON containers(tour_id);

CREATE INDEX IF NOT EXISTS idx_container_details_item ON container_details(item_number);
CREATE INDEX IF NOT EXISTS idx_container_details_location ON container_details(pick_location);
CREATE INDEX IF NOT EXISTS idx_container_details_aisle ON container_details(aisle_sequence);

CREATE INDEX IF NOT EXISTS idx_slotbook_item ON slotbook(item_number);
CREATE INDEX IF NOT EXISTS idx_slotbook_location ON slotbook(location_id);
CREATE INDEX IF NOT EXISTS idx_slotbook_date ON slotbook(inventory_snapshot_date);

CREATE INDEX IF NOT EXISTS idx_labor_data_date ON labor_data(date);
CREATE INDEX IF NOT EXISTS idx_labor_data_hour ON labor_data(hour);
CREATE INDEX IF NOT EXISTS idx_labor_data_wh_date ON labor_data(wh_id, date);

CREATE INDEX IF NOT EXISTS idx_executions_status ON executions(status);
CREATE INDEX IF NOT EXISTS idx_executions_warehouse ON executions(wh_id);
CREATE INDEX IF NOT EXISTS idx_executions_time_range ON executions(start_time, end_time);

-- Performance indexes for new TF/TA tables
CREATE INDEX IF NOT EXISTS idx_tf_container_target_datetime ON tf_container_target(execution_id, planning_datetime);
CREATE INDEX IF NOT EXISTS idx_tf_container_slack_datetime ON tf_container_slack(execution_id, planning_datetime);
CREATE INDEX IF NOT EXISTS idx_tf_container_slack_category ON tf_container_slack(slack_category);

CREATE INDEX IF NOT EXISTS idx_tf_clustering_datetime ON tf_container_clustering(execution_id, planning_datetime);
CREATE INDEX IF NOT EXISTS idx_tf_clustering_cluster ON tf_container_clustering(cluster_id);

CREATE INDEX IF NOT EXISTS idx_tf_metadata_datetime ON tf_clustering_metadata(execution_id, planning_datetime);
CREATE INDEX IF NOT EXISTS idx_tf_metadata_cluster ON tf_clustering_metadata(cluster_id);

CREATE INDEX IF NOT EXISTS idx_tf_tours_datetime ON tf_tour_formation(execution_id, planning_datetime);
CREATE INDEX IF NOT EXISTS idx_tf_tours_cluster ON tf_tour_formation(cluster_id);
CREATE INDEX IF NOT EXISTS idx_tf_tours_tour_id ON tf_tour_formation(tour_id);

CREATE INDEX IF NOT EXISTS idx_ready_tours_archived ON ready_to_release_tours(execution_id, archived_flag);
CREATE INDEX IF NOT EXISTS idx_ready_tours_created ON ready_to_release_tours(created_at_datetime);

CREATE INDEX IF NOT EXISTS idx_ta_pending_datetime ON ta_pending_tours_by_aisle(execution_id, planning_datetime);
CREATE INDEX IF NOT EXISTS idx_ta_pending_aisle ON ta_pending_tours_by_aisle(aisle);

CREATE INDEX IF NOT EXISTS idx_ta_releases_datetime ON ta_tours_to_release(execution_id, planning_datetime);
CREATE INDEX IF NOT EXISTS idx_ta_releases_tour ON ta_tours_to_release(tour_id);
CREATE INDEX IF NOT EXISTS idx_ta_releases_container ON ta_tours_to_release(container_id);
CREATE INDEX IF NOT EXISTS idx_ta_releases_item ON ta_tours_to_release(item_number);

CREATE INDEX IF NOT EXISTS idx_ta_metadata_datetime ON ta_allocation_metadata(execution_id, planning_datetime);

CREATE INDEX IF NOT EXISTS idx_flexsim_datetime ON flexsim_inputs(execution_id, planning_datetime);
CREATE INDEX IF NOT EXISTS idx_flexsim_tour ON flexsim_inputs(tour_id);

CREATE INDEX IF NOT EXISTS idx_planning_iterations_execution ON planning_iterations(execution_id);
CREATE INDEX IF NOT EXISTS idx_planning_iterations_datetime ON planning_iterations(planning_datetime);
CREATE INDEX IF NOT EXISTS idx_planning_iterations_type ON planning_iterations(iteration_type);
CREATE INDEX IF NOT EXISTS idx_planning_iterations_status ON planning_iterations(status);

-- Model trigger decisions tracking
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
    PRIMARY KEY (execution_id, planning_datetime, model_type),
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_trigger_decisions_execution ON trigger_decisions(execution_id);
CREATE INDEX IF NOT EXISTS idx_trigger_decisions_datetime ON trigger_decisions(planning_datetime);
CREATE INDEX IF NOT EXISTS idx_trigger_decisions_model ON trigger_decisions(model_type);
CREATE INDEX IF NOT EXISTS idx_trigger_decisions_triggered ON trigger_decisions(triggered);

-