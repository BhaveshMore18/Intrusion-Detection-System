"""
generate_dataset.py

Produces a labeled training set of feature vectors (data/dataset.csv) using
the SAME feature schema as src/feature_extraction.py.

Why synthetic data? Real malware corpora (EMBER, VirusShare, etc.) are
gigabytes and require registration/licensing to download, which doesn't work
in a portable, from-scratch repo. This generator instead samples benign and
malicious feature vectors from distributions that reflect real, well
documented static-analysis signal, e.g.:
  - Packed/encrypted malware -> higher byte entropy, more high-byte content
  - Malware often ships as PE files with fewer/odd sections, small import
    tables (packed payloads resolve imports at runtime), and at least one
    high-entropy section
  - Benign software tends toward moderate entropy, larger import tables,
    and a "normal" section layout

Swap this file out for a real feature-extraction pass over a labeled corpus
(via src/feature_extraction.py) once you have one -- train.py doesn't care
where dataset.csv comes from as long as the columns match.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_extraction import FEATURE_NAMES  # noqa: E402

RNG = np.random.default_rng(seed=42)


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def generate_benign(n: int) -> pd.DataFrame:
    # Two benign sub-populations:
    #   (a) "plain" files -- scripts, text, uncompressed docs, source code,
    #       moderate entropy.
    #   (b) "compressed everyday files" -- JPG/PNG/PDF/ZIP/MP4/DOCX etc.
    #       These are compressed by their format, NOT malicious, and their
    #       byte entropy is just as high as packed malware (7.5-8.0).
    #       Omitting this sub-population is the #1 cause of false positives
    #       on ordinary photos/PDFs/archives -- ~35% of benign traffic here
    #       reflects that reality.
    n_compressed = int(n * 0.35)
    n_plain = n - n_compressed

    plain_file_size = RNG.lognormal(mean=12.5, sigma=1.3, size=n_plain)
    plain_entropy = RNG.normal(loc=5.2, scale=0.7, size=n_plain).clip(0, 7.2)
    plain_printable = _clip01(RNG.normal(0.55, 0.15, n_plain))
    plain_null = _clip01(RNG.beta(1.5, 6, n_plain))
    plain_high_byte = _clip01(RNG.normal(0.18, 0.08, n_plain))

    comp_file_size = RNG.lognormal(mean=11.5, sigma=1.6, size=n_compressed)
    comp_entropy = RNG.normal(loc=7.75, scale=0.18, size=n_compressed).clip(7.0, 8.0)
    comp_printable = _clip01(RNG.normal(0.35, 0.08, n_compressed))
    comp_null = _clip01(RNG.beta(1.0, 30, n_compressed))
    comp_high_byte = _clip01(RNG.normal(0.48, 0.06, n_compressed))

    file_size = np.concatenate([plain_file_size, comp_file_size])
    byte_entropy = np.concatenate([plain_entropy, comp_entropy])
    printable_ratio = np.concatenate([plain_printable, comp_printable])
    null_byte_ratio = np.concatenate([plain_null, comp_null])
    high_byte_ratio = np.concatenate([plain_high_byte, comp_high_byte])

    # Shuffle the two sub-populations together
    perm = RNG.permutation(n)
    file_size, byte_entropy = file_size[perm], byte_entropy[perm]
    printable_ratio, null_byte_ratio = printable_ratio[perm], null_byte_ratio[perm]
    high_byte_ratio = high_byte_ratio[perm]

    # PE-specific fields: legitimate PE files are a minority of benign
    # traffic and are NOT the high-entropy compressed group (a compressed
    # everyday file like a JPG/PDF is never itself a PE).
    is_pe = RNG.binomial(1, 0.35, n)
    num_sections = np.where(is_pe, RNG.integers(4, 9, n), 0)
    num_imports = np.where(is_pe, RNG.integers(40, 250, n), 0)
    num_suspicious_sections = np.zeros(n, dtype=int)
    has_high_entropy_section = np.where(is_pe, RNG.binomial(1, 0.08, n), 0)
    entry_point_outside = np.where(is_pe, RNG.binomial(1, 0.03, n), 0)
    # A legitimate PE forces overall entropy back to a normal range --
    # executables aren't compressed the way media files are.
    byte_entropy = np.where(is_pe.astype(bool), RNG.normal(5.0, 0.6, n).clip(0, 6.5), byte_entropy)

    df = pd.DataFrame({
        "file_size": file_size,
        "byte_entropy": byte_entropy,
        "printable_ratio": printable_ratio,
        "null_byte_ratio": null_byte_ratio,
        "high_byte_ratio": high_byte_ratio,
        "is_pe": is_pe,
        "num_sections": num_sections,
        "num_imports": num_imports,
        "num_suspicious_sections": num_suspicious_sections,
        "has_high_entropy_section": has_high_entropy_section,
        "entry_point_outside_first_section": entry_point_outside,
    })
    df["label"] = 0
    return df


def generate_malicious(n: int) -> pd.DataFrame:
    file_size = RNG.lognormal(mean=10.8, sigma=1.6, size=n)
    byte_entropy = RNG.normal(loc=7.2, scale=0.5, size=n).clip(0, 8)   # packed => near 8
    printable_ratio = _clip01(RNG.normal(0.22, 0.12, n))
    null_byte_ratio = _clip01(RNG.beta(1.2, 10, n))
    high_byte_ratio = _clip01(RNG.normal(0.45, 0.15, n))

    is_pe = RNG.binomial(1, 0.75, n)
    num_sections = np.where(is_pe, RNG.integers(2, 6, n), 0)
    num_imports = np.where(is_pe, RNG.integers(0, 40, n), 0)          # packed => few resolved imports
    num_suspicious_sections = np.where(is_pe, RNG.binomial(2, 0.25, n), 0)
    has_high_entropy_section = np.where(is_pe, RNG.binomial(1, 0.7, n), 0)
    entry_point_outside = np.where(is_pe, RNG.binomial(1, 0.35, n), 0)

    df = pd.DataFrame({
        "file_size": file_size,
        "byte_entropy": byte_entropy,
        "printable_ratio": printable_ratio,
        "null_byte_ratio": null_byte_ratio,
        "high_byte_ratio": high_byte_ratio,
        "is_pe": is_pe,
        "num_sections": num_sections,
        "num_imports": num_imports,
        "num_suspicious_sections": num_suspicious_sections,
        "has_high_entropy_section": has_high_entropy_section,
        "entry_point_outside_first_section": entry_point_outside,
    })
    df["label"] = 1
    return df


def main(n_benign: int = 3000, n_malicious: int = 3000, out_path: str = None):
    benign = generate_benign(n_benign)
    malicious = generate_malicious(n_malicious)

    df = pd.concat([benign, malicious], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
    df = df[FEATURE_NAMES + ["label"]]  # enforce column order/schema

    out_path = out_path or os.path.join(os.path.dirname(__file__), "dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows ({n_benign} benign / {n_malicious} malicious) -> {out_path}")


if __name__ == "__main__":
    main()
