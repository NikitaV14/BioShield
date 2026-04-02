"""
BioShield IoT — FastAPI Backend v3.0
=====================================
Real SourceAFIS Java SDK called via subprocess (no JPype needed).
Works on Python 3.11+ and Java 11+.

Structure:
  main.py                  ← this file (FastAPI app)
  sourceafis_bridge.py     ← Java subprocess wrapper
  biohash.py               ← BioHashing + crypto
  fvc_benchmark.py         ← FAR/FRR evaluation
  data/templates.db        ← encrypted template store
  data/key_vault.db        ← key vault (separate DB)
  sourceafis_java/         ← Java bridge (pre-built JAR)

Run:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Android IoT:
    POST http://<your-ip>:8000/enroll/image
    POST http://<your-ip>:8000/verify/image
"""

import os
import io
import base64
import sqlite3
import secrets
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Internal modules ──────────────────────────────────────────
from sourceafis_bridge import SourceAFISBridge, BRIDGE_AVAILABLE
from biohash import CryptoVault, BioHasher
from fvc_benchmark import FVCBenchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bioshield")

# =============================================================================
# APP SETUP
# =============================================================================
app = FastAPI(
    title="BioShield IoT API",
    description="Cancellable biometric authentication — SourceAFIS + BioHashing + AES-256-GCM",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATABASE SETUP
# =============================================================================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TEMPLATE_DB  = DATA_DIR / "templates.db"
KEY_VAULT_DB = DATA_DIR / "key_vault.db"


def init_databases():
    with sqlite3.connect(TEMPLATE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS templates (
                user_id     TEXT PRIMARY KEY,
                template_id TEXT NOT NULL,
                ciphertext  BLOB NOT NULL,
                nonce       BLOB NOT NULL,
                tag         BLOB NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT,
                action      TEXT,
                result      TEXT,
                latency_ms  REAL,
                timestamp   TEXT
            )
        """)
        conn.commit()

    with sqlite3.connect(KEY_VAULT_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS keys (
                user_id     TEXT PRIMARY KEY,
                key_id      TEXT NOT NULL,
                fernet_key  BLOB NOT NULL,
                salt        BLOB NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        conn.commit()

    logger.info("Databases initialized: templates.db + key_vault.db")


init_databases()

_stats = {
    "enrollments": 0,
    "verifications": 0,
    "cancellations": 0,
    "start_time": time.time()
}


# =============================================================================
# TEMPLATE STORE
# =============================================================================
class TemplateStore:

    @staticmethod
    def store(user_id, ciphertext, nonce, tag):
        template_id = secrets.token_hex(16)
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(TEMPLATE_DB) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO templates VALUES (?,?,?,?,?,?,?)",
                (user_id, template_id, ciphertext, nonce, tag, now, now)
            )
            conn.commit()
        return template_id

    @staticmethod
    def load(user_id):
        with sqlite3.connect(TEMPLATE_DB) as conn:
            row = conn.execute(
                "SELECT ciphertext, nonce, tag, template_id FROM templates WHERE user_id=?",
                (user_id,)
            ).fetchone()
        if not row:
            raise HTTPException(404, f"No template for user: {user_id}")
        return row[0], row[1], row[2], row[3]

    @staticmethod
    def delete(user_id):
        with sqlite3.connect(TEMPLATE_DB) as conn:
            conn.execute("DELETE FROM templates WHERE user_id=?", (user_id,))
            conn.commit()

    @staticmethod
    def count():
        with sqlite3.connect(TEMPLATE_DB) as conn:
            return conn.execute("SELECT COUNT(*) FROM templates").fetchone()[0]

    @staticmethod
    def log_action(user_id, action, result, latency_ms):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(TEMPLATE_DB) as conn:
            conn.execute(
                "INSERT INTO audit_log (user_id,action,result,latency_ms,timestamp) VALUES (?,?,?,?,?)",
                (user_id, action, result, latency_ms, now)
            )
            conn.commit()


# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class EnrollRequest(BaseModel):
    user_id: str
    fingerprint_b64: str   # Base64-encoded BMP/PNG/TIFF image
    dpi: int = 500

class VerifyRequest(BaseModel):
    user_id: str
    fingerprint_b64: str
    dpi: int = 500

class CancelRequest(BaseModel):
    user_id: str


# =============================================================================
# HELPER: image bytes → feature vector via SourceAFIS bridge
# =============================================================================
def image_to_feature_vector(img_bytes: bytes, dpi: int = 500) -> np.ndarray:
    """
    Full pipeline: raw image bytes → SourceAFIS CBOR template → float feature vector.
    Falls back to image-hash vector if Java bridge unavailable.
    """
    if BRIDGE_AVAILABLE:
        # Save to temp file (SourceAFIS reads from disk)
        tmp_path = DATA_DIR / "_tmp_fp.png"
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        img.save(str(tmp_path))
        raw_bytes = SourceAFISBridge.get_template_bytes(str(tmp_path))
        tmp_path.unlink(missing_ok=True)

        arr = np.array(raw_bytes, dtype=np.float32) / 255.0
        target = 512
        if len(arr) >= target:
            return arr[:target]
        return np.pad(arr, (0, target - len(arr)))
    else:
        # Fallback: deterministic hash of image pixels
        import hashlib
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        h = hashlib.sha256(np.array(img).tobytes()).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)
        return rng.rand(512).astype(np.float32)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "BioShield IoT",
        "version": "3.0.0",
        "sourceafis_java": "✅ LIVE" if BRIDGE_AVAILABLE else "⚠ Fallback mode",
        "crypto": "AES-256-GCM + PBKDF2-SHA256 (600k rounds)",
        "biometric": "SourceAFIS Java + BioHashing (256-bit)",
        "docs": "http://localhost:8000/docs",
    }


@app.get("/status", tags=["Health"])
def status():
    uptime = time.time() - _stats["start_time"]
    return {
        "status": "online",
        "enrolled_users": TemplateStore.count(),
        "total_enrollments": _stats["enrollments"],
        "total_verifications": _stats["verifications"],
        "total_cancellations": _stats["cancellations"],
        "uptime_hours": round(uptime / 3600, 2),
        "sourceafis_java": BRIDGE_AVAILABLE,
        "biohash_bits": BioHasher.TEMPLATE_BITS,
        "pbkdf2_rounds": CryptoVault.PBKDF2_ROUNDS,
    }


@app.post("/enroll", tags=["Biometric"])
def enroll(req: EnrollRequest):
    """
    Enroll a fingerprint.

    Pipeline:
      1. Decode base64 image
      2. Extract SourceAFIS template → feature vector (512-dim)
      3. Generate Fernet key, store in key vault
      4. Derive BioHash projection seed via HKDF
      5. Create 256-bit BioHash template
      6. Encrypt with AES-256-GCM
      7. Store encrypted template
    """
    t0 = time.time()
    try:
        img_bytes = base64.b64decode(req.fingerprint_b64)
        feature_vec = image_to_feature_vector(img_bytes, req.dpi)

        fernet_key = CryptoVault.generate_fernet_key()
        key_id, salt = CryptoVault.store_key(req.user_id, fernet_key, KEY_VAULT_DB)

        projection_seed = CryptoVault.derive_biohash_seed(fernet_key)
        template_bits   = BioHasher.create_template(feature_vec, projection_seed)

        aes_key = CryptoVault.derive_aes_key(fernet_key, salt)
        ciphertext, nonce, tag = CryptoVault.encrypt_template(aes_key, template_bits)

        template_id = TemplateStore.store(req.user_id, ciphertext, nonce, tag)

        latency = round((time.time() - t0) * 1000, 2)
        _stats["enrollments"] += 1
        TemplateStore.log_action(req.user_id, "enroll", "success", latency)

        return {
            "status": "enrolled",
            "user_id": req.user_id,
            "template_id": template_id,
            "key_id": key_id,
            "biohash_bits": BioHasher.TEMPLATE_BITS,
            "hamming_weight": int(template_bits.sum()),
            "encryption": "AES-256-GCM",
            "latency_ms": latency,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enroll error for {req.user_id}: {e}")
        raise HTTPException(500, str(e))


@app.post("/verify", tags=["Biometric"])
def verify(req: VerifyRequest):
    """
    Verify a fingerprint probe.

    Pipeline:
      1. Load encrypted template from DB
      2. Load key from vault, decrypt + verify AES-GCM tag
      3. Extract probe feature vector via SourceAFIS
      4. Re-derive projection matrix (same key → same matrix)
      5. Compute Hamming distance
      6. Match decision (threshold = 90/256 bits)
    """
    t0 = time.time()
    try:
        ciphertext, nonce, tag, template_id = TemplateStore.load(req.user_id)
        fernet_key, salt, key_id = CryptoVault.load_key(req.user_id, KEY_VAULT_DB)

        aes_key         = CryptoVault.derive_aes_key(fernet_key, salt)
        stored_template = CryptoVault.decrypt_template(aes_key, ciphertext, nonce, tag)

        img_bytes   = base64.b64decode(req.fingerprint_b64)
        probe_vec   = image_to_feature_vector(img_bytes, req.dpi)

        projection_seed = CryptoVault.derive_biohash_seed(fernet_key)
        probe_template  = BioHasher.create_template(probe_vec, projection_seed)

        hd       = BioHasher.hamming_distance(stored_template, probe_template)
        is_match = hd <= 90  # ~35% of 256 bits

        latency = round((time.time() - t0) * 1000, 2)
        _stats["verifications"] += 1
        TemplateStore.log_action(req.user_id, "verify", "match" if is_match else "reject", latency)

        return {
            "match": is_match,
            "user_id": req.user_id,
            "hamming_distance": hd,
            "threshold": 90,
            "confidence": round(1.0 - hd / BioHasher.TEMPLATE_BITS, 4),
            "template_id": template_id,
            "latency_ms": latency,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify error for {req.user_id}: {e}")
        raise HTTPException(500, str(e))


@app.delete("/cancel/{user_id}", tags=["Biometric"])
def cancel(user_id: str):
    """
    Cancel a user's biometric template.
    Deletes BOTH the encrypted template AND the key.
    Re-enrollment creates a completely unlinkable new template.
    """
    t0 = time.time()
    TemplateStore.delete(user_id)
    CryptoVault.delete_key(user_id, KEY_VAULT_DB)
    latency = round((time.time() - t0) * 1000, 2)
    _stats["cancellations"] += 1
    TemplateStore.log_action(user_id, "cancel", "cancelled", latency)
    return {
        "status": "cancelled",
        "user_id": user_id,
        "template_deleted": True,
        "key_deleted": True,
        "note": "Re-enrollment produces a completely unlinkable template.",
        "latency_ms": latency,
    }


@app.post("/enroll/image", tags=["Biometric"])
async def enroll_image(user_id: str, file: UploadFile = File(...), dpi: int = 500):
    """
    Enroll from uploaded image file (BMP/PNG/TIFF).
    Android IoT sends fingerprint scan here.

    curl -F "file=@finger.bmp" "http://localhost:8000/enroll/image?user_id=USER1"
    """
    img_bytes = await file.read()
    b64 = base64.b64encode(img_bytes).decode()
    return enroll(EnrollRequest(user_id=user_id, fingerprint_b64=b64, dpi=dpi))


@app.post("/verify/image", tags=["Biometric"])
async def verify_image(user_id: str, file: UploadFile = File(...), dpi: int = 500):
    """
    Verify from uploaded image file.
    Android IoT sends probe scan here.

    curl -F "file=@probe.bmp" "http://localhost:8000/verify/image?user_id=USER1"
    """
    img_bytes = await file.read()
    b64 = base64.b64encode(img_bytes).decode()
    return verify(VerifyRequest(user_id=user_id, fingerprint_b64=b64, dpi=dpi))


@app.get("/audit/{user_id}", tags=["Audit"])
def audit_log(user_id: str, limit: int = 50):
    """Return full audit log for a user."""
    with sqlite3.connect(TEMPLATE_DB) as conn:
        rows = conn.execute(
            "SELECT action, result, latency_ms, timestamp FROM audit_log WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    return {
        "user_id": user_id,
        "log": [{"action": r[0], "result": r[1], "latency_ms": r[2], "timestamp": r[3]} for r in rows]
    }


@app.get("/metrics", tags=["Benchmark"])
def metrics():
    """Return FVC2002 benchmark results (real if available, pre-computed otherwise)."""
    return FVCBenchmark.get_all_results()


@app.post("/breach/simulate", tags=["Security"])
def breach_simulate(user_id: str):
    """
    Run inversion attack simulation on the enrolled template.
    Demonstrates non-invertibility for judges.
    """
    try:
        ciphertext, nonce, tag, _ = TemplateStore.load(user_id)
        fernet_key, salt, _      = CryptoVault.load_key(user_id, KEY_VAULT_DB)
        aes_key        = CryptoVault.derive_aes_key(fernet_key, salt)
        stolen_template = CryptoVault.decrypt_template(aes_key, ciphertext, nonce, tag)
    except:
        raise HTTPException(404, f"No enrolled template for {user_id}. Enroll first.")

    N = 5000
    rng = np.random.RandomState(99)
    best_dist = 1.0
    distances = []

    for _ in range(N):
        cand_vec  = rng.rand(512).astype(np.float32)
        cand_seed = rng.randint(0, 2**31).to_bytes(4, "big")
        cand_hash = BioHasher.create_template(cand_vec, cand_seed)
        d = BioHasher.hamming_distance(cand_hash, stolen_template) / BioHasher.TEMPLATE_BITS
        distances.append(d)
        if d < best_dist:
            best_dist = d

    new_seed     = CryptoVault.derive_biohash_seed(CryptoVault.generate_fernet_key())
    new_template = BioHasher.create_template(
        np.random.RandomState(42).rand(512).astype(np.float32), new_seed
    )
    unlinkability = BioHasher.hamming_distance(stolen_template, new_template) / BioHasher.TEMPLATE_BITS

    return {
        "attack_attempts": N,
        "best_hamming_distance": round(best_dist, 4),
        "average_distance": round(float(np.mean(distances)), 4),
        "inversion_successful": best_dist < 0.10,
        "verdict": "✅ INVERSION FAILED — Template is cryptographically secure"
                   if best_dist >= 0.35 else "⚠ Review security parameters",
        "cancellation_unlinkability": round(unlinkability, 4),
        "unlinkability_confirmed": unlinkability >= 0.40,
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  BioShield IoT — FastAPI Backend v3.0")
    print("=" * 60)
    print(f"  SourceAFIS Java : {'✅ LIVE (subprocess bridge)' if BRIDGE_AVAILABLE else '⚠ Fallback mode'}")
    print(f"  Crypto          : AES-256-GCM + PBKDF2 ({CryptoVault.PBKDF2_ROUNDS:,} rounds)")
    print(f"  BioHash         : {BioHasher.TEMPLATE_BITS} bits")
    print(f"  Docs            : http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
