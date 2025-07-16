from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os

# AES-256 encryption/decryption functions for Smart Health Monitoring System

def encrypt_data(data, key):
    iv = os.urandom(16)
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data.encode()) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ct = encryptor.update(padded_data) + encryptor.finalize()
    return iv + ct

def decrypt_data(encrypted, key):
    iv = encrypted[:16]
    ct = encrypted[16:]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ct) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    return data.decode()
    # ...existing code...

if __name__ == "__main__":
    import os
    # 32-byte key for AES-256
    key = os.urandom(32)
    sample = "PatientID:1,HR:78,BP:120/80,SpO2:98"
    print("Original:", sample)
    encrypted = encrypt_data(sample, key)
    print("Encrypted (hex):", encrypted.hex())
    decrypted = decrypt_data(encrypted, key)
    print("Decrypted:", decrypted)
