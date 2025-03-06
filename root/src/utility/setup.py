#root/src/utility/setup.py

import os
import sys
import subprocess
import venv

def create_venv():
    venv_path = 'pickplan-env'
    if not os.path.exists(venv_path):
        print("Virtual environment not detected. Creating one...")
        try:
            venv.create(venv_path, with_pip=True)
            print(f"Virtual environment {venv_path} created successfully.")
        except Exception as e:
            print(f"Failed to create virtual environment: {e}")
            sys.exit(1)
    else:
        print("Virtual environment already exists.")
    return venv_path

def get_venv_python(venv_path):
    if sys.platform == "win32":
        python_path = os.path.join(venv_path, 'Scripts', 'python.exe')
    else:
        python_path = os.path.join(venv_path, 'bin', 'python')
    return python_path

def ensure_pip(venv_python):
    try:
        subprocess.check_call([venv_python, "-m", "ensurepip", "--upgrade"])
        subprocess.check_call([venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    except subprocess.CalledProcessError:
        print("Failed to ensure pip is available. Please install pip manually.")
        sys.exit(1)

def install_requirements(venv_python):
    print("Installing required packages...")
    requirements_path = os.path.join('root', 'requirements.txt')
    if not os.path.exists(requirements_path):
        print(f"Error: {requirements_path} not found.")
        print("Please ensure the requirements.txt file exists in the 'root' directory.")
        sys.exit(1)
    try:
        subprocess.check_call([venv_python, "-m", "pip", "install", "-r", requirements_path])
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
        print("Please try to install the packages manually.")
        sys.exit(1)

if __name__ == "__main__":
    venv_path = create_venv()
    venv_python = get_venv_python(venv_path)
    ensure_pip(venv_python)
    install_requirements(venv_python)
    print("Setup completed successfully.")