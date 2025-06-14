import os
import sys
import subprocess
import shutil

def find_python_310():
    """Attempts to find a Python 3.10 executable."""
    version_str = "3.10"
    names_to_try = [f"python{version_str}", f"python{version_str}.exe"]

    # Try 'py' launcher on Windows first
    if sys.platform == "win32":
        try:
            # Check if 'py -3.10' works and reports the correct version
            check_command = ['py', f'-{version_str}', '--version']
            print(f"Checking for Python 3.10 via: {' '.join(check_command)}")
            result = subprocess.run(check_command, capture_output=True, text=True, check=True, encoding='utf-8')
            output = result.stdout + result.stderr # Version might be in stderr
            if version_str in output:
                 # Get the actual executable path used by 'py -3.10'
                 get_path_command = ['py', f'-{version_str}', '-c', 'import sys; print(sys.executable)']
                 print(f"Getting path via: {' '.join(get_path_command)}")
                 result_path = subprocess.run(get_path_command, capture_output=True, text=True, check=True, encoding='utf-8')
                 path = result_path.stdout.strip()
                 if path and os.path.exists(path):
                     print(f"Found Python {version_str} via 'py' launcher: {path}")
                     return path
        except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
            print(f"Info: 'py -{version_str}' check failed or Python 3.10 not found via 'py' launcher ({type(e).__name__}). Trying PATH.")
            pass # Continue to check PATH

    # Check standard names in PATH
    print(f"Checking PATH for {names_to_try}")
    for name in names_to_try:
        path = shutil.which(name)
        if path:
            # Verify the version
            try:
                check_command = [path, '--version']
                print(f"Verifying version for {path} using: {' '.join(check_command)}")
                result = subprocess.run(check_command, capture_output=True, text=True, check=True, encoding='utf-8')
                output = result.stdout + result.stderr # Version might be in stderr
                if version_str in output:
                    print(f"Found Python {version_str} in PATH: {path}")
                    return path
                else:
                    print(f"Info: Found {path} but version ({output.strip()}) is not {version_str}.")
            except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
                 print(f"Info: Verification failed for {path} ({type(e).__name__}).")
                 continue
            except Exception as e:
                 print(f"Warning: Unexpected error verifying {path}: {e}")
                 continue


    print(f"Python {version_str} executable not found via 'py' launcher or in PATH.")
    return None


def create_venv():
    venv_path = 'pickopt-env'
    python_310_executable = find_python_310()

    if not python_310_executable:
        print("Error: Python 3.10 executable not found.")
        print("Please install Python 3.10 and ensure it's available in your PATH")
        print("or via the 'py -3.10' launcher (Windows).")
        sys.exit(1)

    if not os.path.exists(venv_path):
        print(f"Creating virtual environment using: {python_310_executable}")
        try:
            # Use the specific python executable to create the venv
            subprocess.check_call([python_310_executable, '-m', 'venv', venv_path])
            print(f"Virtual environment '{venv_path}' created successfully with Python 3.10.")
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to create virtual environment using {python_310_executable}.")
            print(f"Command failed: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print(f"Output: {e.output}")
            print(f"Stderr: {e.stderr}")
            sys.exit(1)
        except FileNotFoundError:
             print(f"Error: The found Python executable '{python_310_executable}' could not be run.")
             print("Please check your Python 3.10 installation.")
             sys.exit(1)
        except Exception as e:
            print(f"Error: An unexpected error occurred during venv creation: {e}")
            sys.exit(1)
    else:
        # Optional: Add check for existing venv's python version
        print(f"Virtual environment '{venv_path}' already exists.")
    return venv_path

def get_venv_python(venv_path):
    """Gets the path to the python executable within the virtual environment."""
    if sys.platform == "win32":
        python_path = os.path.join(venv_path, 'Scripts', 'python.exe')
    else: # Linux/macOS
        python_path = os.path.join(venv_path, 'bin', 'python')

    if not os.path.exists(python_path):
        print(f"Error: Python executable not found at expected path: {python_path}")
        print("The virtual environment might be corrupted or wasn't created correctly.")
        # Attempt to find python3 as a fallback inside bin/Scripts
        fallback_path = os.path.join(os.path.dirname(python_path), 'python3')
        if sys.platform != "win32" and os.path.exists(fallback_path):
             print(f"Found 'python3' instead at: {fallback_path}")
             return fallback_path
        elif sys.platform == "win32":
             fallback_path_py = os.path.join(os.path.dirname(python_path), 'py.exe')
             if os.path.exists(fallback_path_py):
                  print(f"Found 'py.exe' instead at: {fallback_path_py} - Using this, but it might not be the venv python.")
                  return fallback_path_py

        sys.exit(1)
    return python_path

def ensure_pip(venv_python):
    try:
        subprocess.check_call([venv_python, "-m", "ensurepip", "--upgrade"])
        subprocess.check_call([venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    except subprocess.CalledProcessError:
        print("Failed to ensure pip is available. Please install pip manually.")
        sys.exit(1)

def install_requirements(venv_python):
    print("\nInstalling required packages...")
    # Determine the project root directory (two levels up from this script's directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # Pick optimization requirements
    pick_opt_requirements_path = os.path.join(project_root, 'pick_optimization', 'requirements.txt')
    
    # Check pick optimization requirements
    if not os.path.exists(pick_opt_requirements_path):
        print(f"Error: Pick optimization requirements file not found at: {pick_opt_requirements_path}")
        print("Please ensure the requirements.txt file exists in the pick_optimization directory.")
        sys.exit(1)
    
    print(f"Found pick optimization requirements at: {pick_opt_requirements_path}")
    
    # Install requirements file
    try:
        print(f"\nInstalling pick optimization packages...")
        command = [venv_python, "-m", "pip", "install", "-r", pick_opt_requirements_path]
        print(f"Running command: {' '.join(command)}")
        subprocess.check_call(command)
        print(f"Pick optimization packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Failed to install pick optimization packages using command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        # Attempt to capture and print output/error streams if available
        if hasattr(e, 'output') and e.output:
            print(f"Output:\n{e.output}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Stderr:\n{e.stderr}")
        print("\nPlease check the error message above for pick optimization requirements.")
        print("You might need to install the packages manually within the virtual environment:")
        print(f"  1. Activate the environment (e.g., '{os.path.join(os.path.dirname(venv_python), 'activate')}')")
        print(f"  2. Run: pip install -r \"{pick_opt_requirements_path}\"")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\nError: Could not execute pip using '{venv_python}'.")
        print("Please ensure the virtual environment and pip are set up correctly.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: An unexpected error occurred during package installation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("--- Environment Setup Script ---")
    venv_path = create_venv()
    venv_python = get_venv_python(venv_path)
    print(f"Using virtual environment Python: {venv_python}")
    ensure_pip(venv_python)
    install_requirements(venv_python)
    print("Setup completed successfully.")