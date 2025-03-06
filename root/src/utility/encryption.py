#root/src/utility/encryption.py

from cryptography.fernet import Fernet

def generate_key():
    """Generate a key for encryption."""
    return Fernet.generate_key()

def encrypt_credentials(key, credentials):
    """Encrypt the credentials using the provided key."""
    cipher_suite = Fernet(key)
    encrypted_credentials = cipher_suite.encrypt(credentials.encode())
    return encrypted_credentials.decode()

def write_secrets(file_path, key, encrypted_credentials):
    """Write the key and encrypted credentials to a file."""
    with open(file_path, 'w') as file:
        file.write(key.decode() + '\n')
        file.write(encrypted_credentials + '\n')

# Your credentials to encrypt
credentials = "{password}" #Enter pw here

# Generate a key
key = generate_key()

# Encrypt the credentials
encrypted_credentials = encrypt_credentials(key, credentials)

# File path where you want to save the key and encrypted credentials
file_path = 'C:/Users/{username}/cred.txt'

# Write the key and encrypted credentials to the file
write_secrets(file_path, key, encrypted_credentials)

print("Credentials encrypted and written to file.")
