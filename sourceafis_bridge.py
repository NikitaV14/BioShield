"""
sourceafis_bridge.py
====================
Calls the SourceAFIS Java SDK via subprocess.
No JPype, no JVM embedding — just clean process calls.

The JAR is the fat JAR built with Maven in sourceafis_java/target/.
"""

import subprocess
import os
from pathlib import Path

# ── Locate the JAR ────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent

# Primary: Maven-built fat JAR
_JAR_CANDIDATES = [
    _HERE / "sourceafis_java" / "target" / "fingerprint-bridge-1.0-jar-with-dependencies.jar",
    # Fallback: env var override
    Path(os.environ.get("SOURCEAFIS_JAR", "nonexistent")),
]

JAR_PATH = None
for _c in _JAR_CANDIDATES:
    if _c.exists():
        JAR_PATH = str(_c)
        break

BRIDGE_AVAILABLE = JAR_PATH is not None

if BRIDGE_AVAILABLE:
    print(f"[SourceAFIS] Bridge ready: {JAR_PATH}")
else:
    print("[SourceAFIS] JAR not found — running in fallback mode.")
    print("  Build it with: cd sourceafis_java && mvn clean package -q")


def _run_java(args: list[str]) -> list[str]:
    """
    Run the Java bridge and return non-SLF4J output lines.
    """
    result = subprocess.run(
        ["java", "-jar", JAR_PATH] + args,
        capture_output=True, text=True, timeout=30
    )
    lines = [
        l for l in result.stdout.strip().splitlines()
        if not l.startswith("SLF4J")
    ]
    return lines


class SourceAFISBridge:

    @staticmethod
    def get_score(image_path1: str, image_path2: str) -> float:
        """
        Return SourceAFIS similarity score between two fingerprint images.
        Score >= 40 = match. Genuine pairs typically score 80–200.
        """
        if not BRIDGE_AVAILABLE:
            raise RuntimeError("SourceAFIS JAR not found.")
        lines = _run_java(["score", image_path1, image_path2])
        try:
            return float(lines[-1])
        except (IndexError, ValueError):
            return 0.0

    @staticmethod
    def get_template_bytes(image_path: str) -> list[int]:
        """
        Extract SourceAFIS CBOR template bytes from an image.
        Returns a list of uint8 integers (the raw serialized template).
        """
        if not BRIDGE_AVAILABLE:
            raise RuntimeError("SourceAFIS JAR not found.")
        lines = _run_java(["template", image_path])
        try:
            return [int(x) for x in lines[-1].split(",")]
        except (IndexError, ValueError):
            return [0] * 319  # typical template size
