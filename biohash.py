"""
biohash.py
==========
BioHashing (Teoh et al., 2004) + AES-256-GCM cryptography.

Security properties:
  - Non-invertibility : binary projection is one-way
  - Revocability      : change key → completely different template
  - Unlinkability     : different keys → uncorrelated templates (Hamming ~50%)
"""

import os
import sqlite3
import hashlib
from datetime import datetime, timezone

import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend


class CryptoVault:
    """
    All cryptographic operations for BioShield.

    Key hierarchy:
      Fernet key (random, stored in key_vault.db)
        └── HKDF → projection_seed (for BioHash matrix)
        └── PBKDF2 + salt → AES-256 key (for template encryption)
    """

    PBKDF2_ROUNDS = 600_000   # NIST SP 800-132 recommended minimum (2024)
    AES_KEY_SIZE  = 32        # 256-bit AES

    @staticmethod
    def generate_fernet_key() -> bytes:
        return Fernet.generate_key()

    @staticmethod
    def derive_aes_key(fernet_key: bytes, salt: bytes) -> bytes:
        """Derive a 256-bit AES key via PBKDF2-SHA256."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=CryptoVault.AES_KEY_SIZE,
            salt=salt,
            iterations=CryptoVault.PBKDF2_ROUNDS,
            backend=default_backend(),
        )
        return kdf.derive(fernet_key)

    @staticmethod
    def derive_biohash_seed(fernet_key: bytes) -> bytes:
        """
        Derive a 32-byte deterministic seed for the BioHash projection matrix
        using HKDF-SHA256. Same key always yields the same matrix.
        """
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"bioshield-projection-matrix-v3",
            backend=default_backend(),
        )
        return hkdf.derive(fernet_key)

    @staticmethod
    def encrypt_template(aes_key: bytes, template_bits: np.ndarray):
        """Encrypt binary BioHash template with AES-256-GCM."""
        aesgcm = AESGCM(aes_key)
        nonce  = os.urandom(12)   # 96-bit NIST recommended
        ct_tag = aesgcm.encrypt(nonce, template_bits.tobytes(), None)
        return ct_tag[:-16], nonce, ct_tag[-16:]   # ciphertext, nonce, tag

    @staticmethod
    def decrypt_template(aes_key: bytes, ciphertext: bytes, nonce: bytes, tag: bytes) -> np.ndarray:
        """Decrypt and verify AES-GCM tag."""
        aesgcm  = AESGCM(aes_key)
        plain   = aesgcm.decrypt(nonce, ciphertext + tag, None)
        return np.frombuffer(plain, dtype=np.uint8)

    @staticmethod
    def store_key(user_id: str, fernet_key: bytes, key_vault_db) -> tuple:
        key_id = __import__("secrets").token_hex(16)
        salt   = os.urandom(32)
        now    = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(key_vault_db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO keys VALUES (?,?,?,?,?)",
                (user_id, key_id, fernet_key, salt, now)
            )
            conn.commit()
        return key_id, salt

    @staticmethod
    def load_key(user_id: str, key_vault_db) -> tuple:
        with sqlite3.connect(key_vault_db) as conn:
            row = conn.execute(
                "SELECT fernet_key, salt, key_id FROM keys WHERE user_id=?",
                (user_id,)
            ).fetchone()
        if not row:
            from fastapi import HTTPException
            raise HTTPException(404, f"No key found for user: {user_id}")
        return row[0], row[1], row[2]

    @staticmethod
    def delete_key(user_id: str, key_vault_db):
        with sqlite3.connect(key_vault_db) as conn:
            conn.execute("DELETE FROM keys WHERE user_id=?", (user_id,))
            conn.commit()


class BioHasher:
    """
    BioHashing: projects a feature vector onto a random subspace
    (seeded from the user's secret key) and binarises at zero.

    Result: 256-bit binary template that is:
      - Non-invertible (quantization is one-way)
      - Revocable      (change key → new uncorrelated template)
      - Unlinkable     (Hamming distance ~50% between any two keys)
    """

    TEMPLATE_BITS = 256

    @staticmethod
    def _projection_matrix(seed: bytes, n_features: int, n_bits: int) -> np.ndarray:
        """
        Generate a random Gaussian projection matrix from seed bytes.
        Uses Gram-Schmidt orthonormalization for optimal theoretical properties.
        Shape: (n_bits, n_features)
        """
        rng    = np.random.default_rng(np.frombuffer(seed, dtype=np.uint32))
        matrix = rng.standard_normal((n_bits, n_features))
        if n_bits <= n_features:
            Q, _ = np.linalg.qr(matrix.T)
            return Q.T[:n_bits]
        return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)

    @classmethod
    def create_template(cls, feature_vector: np.ndarray, projection_seed: bytes) -> np.ndarray:
        """
        Create a binary BioHash template.
        template[i] = 1 if (P[i] · feature_vector) >= 0 else 0
        """
        P    = cls._projection_matrix(projection_seed, len(feature_vector), cls.TEMPLATE_BITS)
        proj = P @ feature_vector.astype(np.float64)
        return (proj >= 0).astype(np.uint8)

    @staticmethod
    def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
        return int(np.sum(a != b))

    @staticmethod
    def is_match(a: np.ndarray, b: np.ndarray, threshold: int = 90) -> bool:
        return BioHasher.hamming_distance(a, b) <= threshold
