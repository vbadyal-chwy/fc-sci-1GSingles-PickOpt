# FC Science Pick Optimization Model

A Python-based optimization model for FC Pick Optimization. 
## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Overview

[Pick Optimization Research Confluence](https://chewyinc.atlassian.net/l/cp/XqjmR7KM)

This project implements a comprehensive pick optimization system with two main components:

- **Simulator**: End-to-end simulation orchestrator that manages data flow, database operations, and coordinates the execution of optimization models
- **Pick Optimization**: Blackbox optimization module containing Tour Formation (TF) and Tour Allocation (TA) algorithms that can be called independently or as part of the simulation

### Architecture

The system follows a clean separation of concerns:

1. **Simulation Layer** (`simulator/`): Handles data extraction, database management, input/output processing, and model orchestration
2. **Optimization Layer** (`pick_optimization/`): Contains the core optimization algorithms and can be used as a standalone module
3. **Clean Interface**: The optimization module can be imported and used independently of the simulation infrastructure 

## Requirements

Before setting up the project, ensure you have the following prerequisites:

- **Python**: 3.10 (or <3.13 for Snowflake compatability)
- **Vertica EDW Access**: Required for data access
- **Gurobi Optimizer**: Required for optimization calculations

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Chewy-Inc/fc-sci-pick-opt-model.git
   ```

2. **Run the Setup Script:**
   ```bash
   python env_setup.py
   ```
   This script creates the virtual environment `pickopt-env` and installs all required packages.

3. **Activate the Virtual Environment:**
   ```bash
   # On Windows:
  .\pickopt-env\Scripts\activate
   ```

## Usage

### **Running the End-to-End Simulation:**
Run the full simulation orchestrator:
```bash
python simulator/end_to_end_simulation.py
```

### **Running Individual Optimization Models:**
For standalone optimization tasks:
```bash
python pick_optimization/main.py
```

### **Running Tests:**
```bash
# Tour Formation tests
python pick_optimization/tf_test.py

# Tour Allocation tests
python pick_optimization/ta_test.py
```

## Project Structure

```plaintext
fc-sci-pick-opt-model/
├── simulator/                          # Simulation orchestration layer
│   ├── end_to_end_simulation.py       # Main simulation script
│   ├── data_store/                    # Database infrastructure
│   │   ├── create_database.py         # Database creation utilities
│   │   ├── external_data/             # Data extraction & transformation
│   │   └── core/                      # Core database management
│   ├── data/                          # Simulation data files
│   └── utils/                         # Simulation-specific utilities
│       ├── database_tf_inputs.py      # TF input generation from database
│       ├── database_ta_inputs.py      # TA input generation from database
│       ├── tf_output_processor.py     # TF output processing
│       ├── ta_output_processor.py     # TA output processing
│       └── model_trigger_logic.py     # Model execution triggers
│
├── pick_optimization/                  # Optimization blackbox module
│   ├── tour_formation/                # Tour formation algorithms
│   ├── tour_allocation/               # Tour allocation logic
│   ├── input/                         # Model input directories
│   ├── output/                        # Model output directories
│   ├── working/                       # Model working directories
│   ├── sql/                           # Optimization SQL queries
│   ├── utils/                         # Optimization-specific utilities
│   │   ├── env_setup.py              # Environment setup
│   │   ├── db_util.py                # Database utilities
│   │   ├── platform_utils.py         # Platform utilities
│   │   └── logging_config.py         # Logging configuration
│   ├── main.py                        # Optimization entry point
│   ├── tf_test.py                     # Tour formation tests
│   ├── ta_test.py                     # Tour allocation tests
│   ├── end_to_end_test.py            # Legacy end-to-end test
│   ├── end_to_end_test.ipynb         # Testing notebook
│   ├── requirements.txt               # Optimization dependencies
│   └── README.md                      # Optimization module README
│
├── data_store/                        # Legacy database files (to be cleaned up)
├── pickopt-env/                       # Virtual environment
├── scripts/                           # Utility scripts
├── gradle/                            # Gradle build files
├── build.gradle                       # Gradle build configuration
├── requirements.txt                   # Root level dependencies
└── README.md                          # This README file
```


