import yaml
import os
from typing import Dict, Any
from cryptography.fernet import Fernet

def read_secrets(file_path: str) -> tuple:
    with open(file_path, 'r') as file:
        key = file.readline().strip()
        encrypted_credentials = file.readline().strip()
    return key, encrypted_credentials

def decrypt_credentials(key: str, encrypted_credentials: str) -> str:
    cipher_suite = Fernet(key)
    decrypted_credentials = cipher_suite.decrypt(encrypted_credentials.encode())
    return decrypted_credentials.decode()

def get_creds(file_path: str) -> tuple:
    key, encrypted_credentials = read_secrets(file_path)
    decrypted_credentials = decrypt_credentials(key, encrypted_credentials)
    username = os.path.basename(os.path.dirname(file_path))
    return username, decrypted_credentials

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Replace ${USER} with the actual username in the credentials file paths
    if 'vertica' in config['database']:
        config['database']['vertica']['credentials_file'] = config['database']['vertica']['credentials_file'].replace('${USER}', os.getenv('USER', ''))
    
    if 'mssql' in config['database'] and 'credentials_file' in config['database']['mssql']:
        config['database']['mssql']['credentials_file'] = config['database']['mssql']['credentials_file'].replace('${USER}', os.getenv('USER', ''))
    
    # Load and decrypt Vertica credentials
    if 'vertica' in config['database']:
        username, password = get_creds(config['database']['vertica']['credentials_file'])
        config['database']['vertica']['username'] = username
        config['database']['vertica']['password'] = password
    
    # Load and decrypt MSSQL credentials if they exist
    if 'mssql' in config['database'] and 'credentials_file' in config['database']['mssql']:
        username, password = get_creds(config['database']['mssql']['credentials_file'])
        config['database']['mssql']['username'] = username
        config['database']['mssql']['password'] = password
    
    # Load Gurobi config
    gurobi_config_path = os.path.join(os.path.dirname(config_path), 'gurobi_config.yaml')
    if os.path.exists(gurobi_config_path):
        with open(gurobi_config_path, 'r') as gurobi_config_file:
            config['gurobi'] = yaml.safe_load(gurobi_config_file)
    else:
        print(f"Warning: Gurobi config file not found at {gurobi_config_path}")
    
    return config

# Usage example
if __name__ == "__main__":
    config = load_config('path/to/your/config.yaml')
    print("Configuration loaded successfully")
    if 'vertica' in config['database']:
        print(f"Vertica username: {config['database']['vertica']['username']}")
        print(f"Vertica password: {'*' * len(config['database']['vertica']['password'])}")
    if 'mssql' in config['database'] and 'username' in config['database']['mssql']:
        print(f"MSSQL username: {config['database']['mssql']['username']}")
        print(f"MSSQL password: {'*' * len(config['database']['mssql']['password'])}")
    if 'gurobi' in config:
        print("Gurobi configuration loaded")
    else:
        print("Gurobi configuration not found")