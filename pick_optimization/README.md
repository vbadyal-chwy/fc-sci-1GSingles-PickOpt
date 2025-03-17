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

## Requirements

Before setting up the project, ensure you have the following prerequisites:

- **Python**: 3.12 or higher
- **Vertica EDW Access**: Required for data access
- **Gurobi Optimizer**: Required for optimization calculations

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Chewy-Inc/fc-sci-pick-planning-model.git
   ```

2. **Run the Setup Script:**
   ```bash
   python setup.py
   ```
   This script creates the virtual environment `pickplan-env` and installs all required packages.

3. **Activate the Virtual Environment:**
   ```bash
   # On Windows:
   pickplan-env\Scripts\activate
   ```

4. **Configure Environment:**
   - Ensure `config/config.yaml` is properly configured.


## Usage

The model can be run in two ways:

1. **Using Jupyter Notebook:**
   Open and run `main.ipynb` for an interactive experience:
   ```python
   # The notebook provides step-by-step execution:
   # 1. Import libraries and modules
   # 2. Load configurations
   # 3. Load and validate input data
   # 4. Run pick optimization model
   ```

2. **Using Python Script:**
   Run the main script directly:
   ```bash
   python pick_optimization/main.py
   ```

## Project Structure

```plaintext
pick_optimization/
├── config/                 # Configuration files
│   └── config.yaml         # Main configuration file
├── data/                   # Data handling modules
│   ├── data_puller.py      # Data retrieval functionality
│   └── data_validator.py   # Data validation logic
├── engine/                # Core optimization engine
│   └── sim_engine.py      # Sim. engine implementation
├── input/                 # Input data directory
├── output/                # Output results directory
├── tour_allocation/       # Tour allocation logic
├── tour_formation/        # Tour formation algorithms
├── utils/                 # Utility functions
├── main.ipynb            # Interactive notebook
├── main.py               # Main script
├── requirements.txt      # Project dependencies
├── setup.py             # Setup script
└── settings.json        # IDE settings
```

## Output

The model generates optimized tours to release in the `output/` directory.

