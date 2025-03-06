# pick-planning-model
FC Science Pick Optimization Research Model

## Table of Contents
- [Documentation](#documentation)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Documentation
1. [Pick Optimization Research Confluence](https://chewyinc.atlassian.net/l/cp/XqjmR7KM) 

## Requirements

Before setting up the project, ensure you have the following prerequisites:

- **Python**: 3.8 or higher
- **Vertica EDW Access**: [Request access here for Fulfillment, Procurement & Fulfillment Sandbox](https://chewy.service-now.com/sh?id=sc_cat_item_guide&table=sc_cat_item&sys_id=9905bd99db447f803f1f929adb96199f&recordUrl=com.glideapp.servicecatalog_cat_item_guide_view.do%3Fv%3D1&sysparm_initial=true&sysparm_guide=9905bd99db447f803f1f929adb96199f)
- **Disk Space**: At least 2 GB of free hard disk space
- **IDE**: Visual Studio Code (preferred)
- **Gurobi Optimizer Key**: Contact author for the `gurobi_config.yaml` file
- **Git**: For version control
- 
---

## Installation

Follow these steps to install and set up the project:

1. **Clone the Repository:**
  ```
   git clone https://github.com/Chewy-Inc/fc-sci-pick-planning-model.git
  ```

2. **Run the Setup Script:**
   ```
   python setup.py
   ```
   This script creates the virtual environment \`pickplan-env\` (if it doesn't exist) and installs all required packages.

   
3. **Activate the Virtual Environment:**
   - **In IDE (Visual Studio Code):**
     
     1. Press \`Ctrl+Shift+P\` -> Search for \`Workspace settings\` -> Open \`settings.json\`.
     2. Add the following configuration  (also available in settings.json):
       ```plaintext
        {
            "python.defaultInterpreterPath": "${workspaceFolder}\\pickplan-env\\Scripts\\python.exe",
            "python.linting.pylintPath": "${workspaceFolder}\\pickplan-env\\Scripts\\pylint.exe",
            "python.formatting.blackPath": "${workspaceFolder}\\pickplan-env\\Scripts\\black.exe",
            "python.linting.flake8Path": "${workspaceFolder}\\pickplan-env\\Scripts\\flake8.exe"
        }
        ```
   - **In Command Line (CMD/Terminal):**
    ```
     bash
     source pickplan-env/bin/activate   # On Windows: pickplan-env\Scripts\activate
     ```
    
4. **Set up Database Connection Secrets:**
   - Open \`/src/encryption.py\`.
   - Replace \`{username}\` and \`{password}\` with your database credentials.
   - Execute the script to generate an encrypted \`.txt\` file with your credentials.
   - **Important**: Delete the username and password from \`encryption.py\` after execution.

---

## Usage

1. **Run the Main Script:**
   Open and run the Jupyter notebook \`src/main.ipynb\`. You have the ability run one cell(workflow) at a time or execute all cells to run the entire workflow.

---

## Project Structure (TBD)

```plaintext
root
│
├── config/
│   ├── config.yaml
│   └── gurobi_config.yaml
│
│
├── input/
│   ├── sample_input_files(.csv)
|
├── output/
│   ├── sample_output_files(.csv)
|
├── sql/
|
├── src/
│
├── .gitignore
├── requirements.txt
├── README.md
└── settings.json
```
---
