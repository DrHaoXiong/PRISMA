from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


IDENTITY_LD_MESSAGE = (
    "Identity Laplacian was requested or implied without an empirical LD reference. "
    "For real-data analyses, provide --bfile with --ld-reference-mode plink/auto. "
    "To intentionally run diagnostic identity-LD mode, rerun with "
    "--ld-reference-mode identity --allow-identity-ld."
)


@dataclass(frozen=True)
class LDReferenceResolution:
    bfile_path: str | None
    identity_ld_active: bool
    identity_ld_reason: str | None
    warnings: list[str] = field(default_factory=list)


def manifest_allows_auto_identity(manifest_path: str | os.PathLike[str]) -> bool:
    """Allow auto identity-LD only for bundled examples or tests."""
    parts = {part.lower() for part in Path(manifest_path).resolve().parts}
    return "examples" in parts or "tests" in parts


def resolve_ld_reference(
    manifest_path: str | os.PathLike[str],
    ld_reference_mode: str = "auto",
    bfile: str | None = None,
    allow_identity_ld: bool = False,
) -> LDReferenceResolution:
    """Resolve LD-reference policy shared by run_prisma.py and tune_rank.py."""
    if ld_reference_mode == "plink" and not bfile:
        raise ValueError("--ld-reference-mode plink requires --bfile.")
    if ld_reference_mode == "identity" and not allow_identity_ld:
        raise ValueError(IDENTITY_LD_MESSAGE)

    bfile_path = bfile if ld_reference_mode in {"plink", "auto"} and bfile else None
    identity_ld_active = False
    identity_ld_reason = None
    warnings: list[str] = []

    if bfile_path is None:
        if ld_reference_mode == "auto":
            if allow_identity_ld:
                identity_ld_active = True
                identity_ld_reason = "explicit_allow_identity_ld"
                warnings.append(IDENTITY_LD_MESSAGE)
            elif manifest_allows_auto_identity(manifest_path):
                identity_ld_active = True
                identity_ld_reason = "examples_or_tests_manifest"
                warnings.append(
                    f"{IDENTITY_LD_MESSAGE} "
                    "Proceeding because the manifest is under examples/ or tests/."
                )
            else:
                raise ValueError(IDENTITY_LD_MESSAGE)
        elif ld_reference_mode == "identity":
            identity_ld_active = True
            identity_ld_reason = "explicit_identity_mode"
            warnings.append(IDENTITY_LD_MESSAGE)
    else:
        missing_plink = [
            f"{bfile_path}{suffix}"
            for suffix in [".bed", ".bim", ".fam"]
            if not os.path.exists(f"{bfile_path}{suffix}")
        ]
        if missing_plink:
            raise FileNotFoundError(f"Missing PLINK reference files: {missing_plink}")

    return LDReferenceResolution(
        bfile_path=bfile_path,
        identity_ld_active=identity_ld_active,
        identity_ld_reason=identity_ld_reason,
        warnings=warnings,
    )
