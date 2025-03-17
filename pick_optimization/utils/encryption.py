"""
Encryption utilities for pick optimization.

This module provides functions for encrypting and decrypting sensitive data
such as database credentials.
"""
from cryptography.fernet import Fernet
from typing import Optional


def generate_key() -> bytes:
    """
    Generate a key for encryption.
    
    Returns
    -------
    bytes
        A new encryption key
    """
    return Fernet.generate_key()


def encrypt_credentials(key: bytes, credentials: str) -> str:
    """
    Encrypt the credentials using the provided key.
    
    Parameters
    ----------
    key : bytes
        The encryption key
    credentials : str
        The credentials to encrypt
        
    Returns
    -------
    str
        The encrypted credentials as a string
    """
    cipher_suite = Fernet(key)
    encrypted_credentials = cipher_suite.encrypt(credentials.encode())
    return encrypted_credentials.decode()


def write_secrets(file_path: str, key: bytes, encrypted_credentials: str) -> None:
    """
    Write the key and encrypted credentials to a file.
    
    Parameters
    ----------
    file_path : str
        Path to the file where secrets will be written
    key : bytes
        The encryption key
    encrypted_credentials : str
        The encrypted credentials
    """
    with open(file_path, 'w') as file:
        file.write(key.decode() + '\n')
        file.write(encrypted_credentials + '\n')


def create_credentials_file(file_path: str, credentials: str) -> None:
    """
    Create a new credentials file with encrypted credentials.
    
    This is a convenience function that generates a key, encrypts the
    credentials, and writes them to a file.
    
    Parameters
    ----------
    file_path : str
        Path to the file where secrets will be written
    credentials : str
        The credentials to encrypt
    """
    key = generate_key()
    encrypted_credentials = encrypt_credentials(key, credentials)
    write_secrets(file_path, key, encrypted_credentials) 