from pathlib import Path

import pytest

from ld_policy import resolve_ld_reference


def test_auto_identity_allowed_for_examples_manifest():
    manifest = Path("examples/mini_fixture_1000/manifest.csv")
    resolution = resolve_ld_reference(manifest, ld_reference_mode="auto")
    assert resolution.bfile_path is None
    assert resolution.identity_ld_active is True
    assert resolution.identity_ld_reason == "examples_or_tests_manifest"
    assert resolution.warnings


def test_auto_identity_blocks_real_data_without_permission(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("type,name,path\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Identity Laplacian"):
        resolve_ld_reference(manifest, ld_reference_mode="auto")


def test_plink_requires_bfile():
    with pytest.raises(ValueError, match="requires --bfile"):
        resolve_ld_reference("examples/mini_fixture_1000/manifest.csv", ld_reference_mode="plink")
