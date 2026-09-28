"""
feature_extraction.py

Turns a raw file on disk into a fixed-length numeric feature vector that the
neural network can classify as benign or malicious.

Two layers of features:
  1. Generic byte-level statistics -- work on ANY file type (exe, pdf, doc,
     script, etc.). Malware authors frequently pack/encrypt payloads, which
     shows up clearly in byte entropy and byte-value distribution.
  2. PE (Windows executable) structural features -- parsed with `pefile`
     when the file is a valid PE. These are the same signal categories real
     antivirus/EDR static-analysis engines look at (section counts, unusual
     entry points, import counts, suspicious section names, etc.).

If a file isn't a PE, the PE-specific fields are filled with neutral
defaults so the feature vector stays a fixed length regardless of file type.
"""

import math
import os
from collections import Counter

import numpy as np

try:
    import pefile
    PEFILE_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False

# Order matters -- this defines the feature vector layout used everywhere
# else in the project (dataset generation, training, prediction).
FEATURE_NAMES = [
    "file_size",
    "byte_entropy",
    "printable_ratio",
    "null_byte_ratio",
    "high_byte_ratio",       # bytes >= 0x80 -- common in packed/encrypted data
    "is_pe",
    "num_sections",
    "num_imports",
    "num_suspicious_sections",
    "has_high_entropy_section",
    "entry_point_outside_first_section",
]

SUSPICIOUS_SECTION_NAMES = {
    ".packed", ".upx0", ".upx1", ".upx2", ".aspack", ".vmp0", ".vmp1",
    ".themida", ".petite", ".nsp0", ".nsp1", ".enigma",
}


def _shannon_entropy(data: bytes) -> float:
    """Standard Shannon entropy over byte values, 0 (uniform) .. 8 (random)."""
    if not data:
        return 0.0
    counts = Counter(data)
    length = len(data)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _generic_byte_features(data: bytes) -> dict:
    if not data:
        return {
            "file_size": 0,
            "byte_entropy": 0.0,
            "printable_ratio": 0.0,
            "null_byte_ratio": 0.0,
            "high_byte_ratio": 0.0,
        }

    length = len(data)
    printable = sum(1 for b in data if 32 <= b <= 126)
    nulls = data.count(0)
    high = sum(1 for b in data if b >= 0x80)

    return {
        "file_size": length,
        "byte_entropy": _shannon_entropy(data),
        "printable_ratio": printable / length,
        "null_byte_ratio": nulls / length,
        "high_byte_ratio": high / length,
    }


def _pe_features(path: str) -> dict:
    """Parse PE structural features. Returns neutral defaults on any failure
    (non-PE file, corrupt header, pefile not installed)."""
    defaults = {
        "is_pe": 0,
        "num_sections": 0,
        "num_imports": 0,
        "num_suspicious_sections": 0,
        "has_high_entropy_section": 0,
        "entry_point_outside_first_section": 0,
    }

    if not PEFILE_AVAILABLE:
        return defaults

    try:
        pe = pefile.PE(path, fast_load=True)
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
        )
    except Exception:
        return defaults

    result = dict(defaults)
    result["is_pe"] = 1
    result["num_sections"] = len(pe.sections)

    # Count imported functions across all imported DLLs
    num_imports = 0
    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            num_imports += len(entry.imports)
    result["num_imports"] = num_imports

    suspicious = 0
    high_entropy_section = 0
    for section in pe.sections:
        name = section.Name.decode(errors="ignore").strip("\x00").lower()
        if name in SUSPICIOUS_SECTION_NAMES:
            suspicious += 1
        try:
            if section.get_entropy() >= 7.0:
                high_entropy_section = 1
        except Exception:
            pass
    result["num_suspicious_sections"] = suspicious
    result["has_high_entropy_section"] = high_entropy_section

    try:
        entry_point = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        first_section = pe.sections[0]
        in_first = (
            first_section.VirtualAddress
            <= entry_point
            < first_section.VirtualAddress + first_section.Misc_VirtualSize
        )
        result["entry_point_outside_first_section"] = 0 if in_first else 1
    except Exception:
        pass

    pe.close()
    return result


def extract_features(path: str) -> np.ndarray:
    """Extract the full fixed-length feature vector for a single file."""
    with open(path, "rb") as f:
        data = f.read()

    features = {}
    features.update(_generic_byte_features(data))
    features.update(_pe_features(path))

    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)


def extract_features_dict(path: str) -> dict:
    """Same as extract_features but returns a name -> value dict, handy for
    the dashboard where we want to show individual feature readings."""
    vec = extract_features(path)
    return dict(zip(FEATURE_NAMES, vec.tolist()))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python feature_extraction.py <path-to-file>")
        sys.exit(1)

    target = sys.argv[1]
    if not os.path.isfile(target):
        print(f"No such file: {target}")
        sys.exit(1)

    for name, value in extract_features_dict(target).items():
        print(f"{name:35s} {value}")
