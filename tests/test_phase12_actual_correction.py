import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from app.tools.correction_engine import apply_corrections
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_file_sha256(filepath: Path) -> str:
    """Helper to calculate SHA-256 of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def test_actual_correction_immutability():
    """
    Test that running the correction engine on an input dataset DOES NOT modify the original file.
    The SHA-256 hash before and after MUST match.
    """
    raw_dataset = PROJECT_ROOT / "data" / "raw" / "sales_problematic.csv"
    if not raw_dataset.exists():
        pytest.skip(f"Test dataset {raw_dataset} not found.")

    original_hash_before = get_file_sha256(raw_dataset)

    # Execute correction
    df_corrected, exec_result = apply_corrections(
        dataset_input=raw_dataset,
        schema="sales",
        dataset_name="sales_problematic",
    )

    original_hash_after = get_file_sha256(raw_dataset)

    # 1. Verify hashes match
    assert original_hash_before == original_hash_after, "The original file was modified during correction!"

    # 2. Verify corrected file was actually generated
    assert exec_result.corrected_dataset_path is not None
    generated_path = Path(exec_result.corrected_dataset_path)
    assert generated_path.exists(), "Corrected dataset file was not generated."

    # 3. Verify it's not the same path
    assert generated_path.resolve() != raw_dataset.resolve()

    # Cleanup generated test file
    if generated_path.exists():
        generated_path.unlink()
