# BioShield IoT — Backend

FastAPI backend for cancellable biometric authentication using SourceAFIS + BioHashing + AES-256-GCM.

## Quick Start

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: http://localhost:8000/docs

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — all REST endpoints |
| `biohash.py` | BioHashing (Teoh et al., 2004) + AES-256-GCM crypto |
| `sourceafis_bridge.py` | SourceAFIS Java SDK via subprocess (no JPype) |
| `fvc_benchmark.py` | FVC2002 benchmark results + pre-computed metrics |
| `train_evaluate.py` | Full FVC2002 training + evaluation pipeline |

## Key Results

| Metric | Value |
|--------|-------|
| Algorithm | SourceAFIS Java 3.18.1 |
| FVC2002 DB1_B EER | **0.71%** |
| Genuine avg score | 138.34 |
| Impostor avg score | 4.34 |
| Score separation | 134× |
| FAR at threshold 40 | **0.00%** |
| Inversion best distance | 0.3984 (≈ random) |
| Template encryption | AES-256-GCM |
| Key derivation | PBKDF2-SHA256 (600k rounds) |

## Prerequisites

- Python 3.11+
- Java 11+ (for SourceAFIS bridge)
- JAR: `sourceafis_java/target/fingerprint-bridge-1.0-jar-with-dependencies.jar`

## API Endpoints

```
GET  /          → Health check
GET  /status    → System stats
POST /enroll    → Enroll fingerprint
POST /verify    → Verify fingerprint
DELETE /cancel/{user_id} → Cancel template
GET  /audit/{user_id}    → Audit log
GET  /metrics   → FVC2002 benchmark results
POST /breach/simulate    → Inversion attack demo
POST /enroll/image       → Enroll from image file
POST /verify/image       → Verify from image file
```
