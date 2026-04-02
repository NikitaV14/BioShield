# BioShield IoT — Setup Guide

## Project Structure

```
BioShield/
├── main.py                  ← FastAPI backend (start here)
├── sourceafis_bridge.py     ← SourceAFIS Java subprocess wrapper
├── biohash.py               ← BioHashing + AES-256-GCM crypto
├── fvc_benchmark.py         ← FVC2002 results + benchmark module
├── requirements.txt
├── sourceafis_java/         ← Copy from your Downloads folder
│   ├── pom.xml
│   ├── src/main/java/FingerprintBridge.java
│   └── target/
│       └── fingerprint-bridge-1.0-jar-with-dependencies.jar
└── data/                    ← Auto-created on first run
    ├── templates.db         ← Encrypted template store
    └── key_vault.db         ← Key vault (separate DB)
```

---

## Step 1 — Copy the JAR

Copy your already-built JAR into the project:

```
sourceafis_java\target\fingerprint-bridge-1.0-jar-with-dependencies.jar
```

This is the JAR you already built with `mvn clean package -q`.

---

## Step 2 — Install Python dependencies

```bash
bioshield_env\Scripts\activate
pip install -r requirements.txt
```

---

## Step 3 — Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser: http://localhost:8000/docs

You should see the interactive API documentation with all endpoints.

---

## Step 4 — Test the API

### Enroll a fingerprint
```bash
curl -F "file=@C:\Users\nikita\Downloads\archive\fingerprints\DB1_B\101_1.tif" \
     "http://localhost:8000/enroll/image?user_id=user_101"
```

### Verify a fingerprint
```bash
curl -F "file=@C:\Users\nikita\Downloads\archive\fingerprints\DB1_B\101_2.tif" \
     "http://localhost:8000/verify/image?user_id=user_101"
```

### Cancel a template
```bash
curl -X DELETE "http://localhost:8000/cancel/user_101"
```

### Run breach simulation
```bash
curl -X POST "http://localhost:8000/breach/simulate?user_id=user_101"
```

### View metrics
```bash
curl "http://localhost:8000/metrics"
```

---

## Android Studio Integration (IoT)

Your Android app will send fingerprint images to the server over WiFi.

### Android sends to:
```
POST http://<your-PC-IP>:8000/enroll/image?user_id=<id>
POST http://<your-PC-IP>:8000/verify/image?user_id=<id>
```

### Find your PC IP:
```bash
ipconfig
```
Look for IPv4 Address under your WiFi adapter, e.g. `192.168.1.105`

### Android permissions needed in AndroidManifest.xml:
```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.USE_BIOMETRIC" />
```

### Android Retrofit API call (Kotlin):
```kotlin
// Enroll
val file = File(imagePath)
val body = RequestBody.create("image/bmp".toMediaTypeOrNull(), file)
val part = MultipartBody.Part.createFormData("file", file.name, body)
val response = api.enrollImage(userId = "user_101", file = part)

// Verify
val response = api.verifyImage(userId = "user_101", file = part)
// response.match == true → access granted
```

### Retrofit interface:
```kotlin
interface BioShieldApi {
    @Multipart
    @POST("enroll/image")
    suspend fun enrollImage(
        @Query("user_id") userId: String,
        @Part file: MultipartBody.Part
    ): EnrollResponse

    @Multipart
    @POST("verify/image")
    suspend fun verifyImage(
        @Query("user_id") userId: String,
        @Part file: MultipartBody.Part
    ): VerifyResponse
}
```

### Response models (Kotlin data classes):
```kotlin
data class VerifyResponse(
    val match: Boolean,
    val user_id: String,
    val hamming_distance: Int,
    val confidence: Double,
    val latency_ms: Double
)
```

---

## Key Results (for judges)

| Metric | Value |
|--------|-------|
| Algorithm | SourceAFIS Java 3.18.1 |
| FVC2002 DB1_B EER | **0.71%** |
| Genuine avg score | 138.34 |
| Impostor avg score | 4.34 |
| Score separation | 134x |
| FAR at threshold 40 | **0.00%** |
| FRR at threshold 40 | 7.14% |
| Inversion best distance | 0.3984 (≈ random) |
| Unlinkability distance | 0.5234 |
| Template encryption | AES-256-GCM |
| Key derivation | PBKDF2-SHA256 (600k rounds) |
