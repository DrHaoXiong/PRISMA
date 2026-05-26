#!/usr/bin/env python3
"""Generate a deterministic 1000-SNP synthetic PRISMA mini-fixture."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


TISSUES = ["retina", "blood", "artery"]


def two_sided_p_from_z(z: np.ndarray) -> np.ndarray:
    return np.array([math.erfc(abs(float(x)) / math.sqrt(2.0)) for x in z])


def generate_fixture(out_dir: Path, seed: int = 20260526, n_snps: int = 1000) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    rel_out = out_dir.relative_to(repo_root)

    snps = np.array([f"rsMINI{i:06d}" for i in range(1, n_snps + 1)])
    chrom = np.repeat(np.arange(1, 6), math.ceil(n_snps / 5))[:n_snps]
    within_chrom = np.tile(np.arange(math.ceil(n_snps / 5)), 5)[:n_snps]
    bp = 1_000_000 + within_chrom * 2_000

    allele_pairs = np.array([["A", "C"], ["A", "G"], ["C", "T"], ["G", "T"]])
    allele_idx = rng.integers(0, len(allele_pairs), n_snps)
    a1 = allele_pairs[allele_idx, 0]
    a2 = allele_pairs[allele_idx, 1]

    axis = np.arange(n_snps) % 3
    latent = rng.normal(0, 1, n_snps)
    gwas_z = latent + np.choose(axis, [1.4, 1.0, 0.8]) + rng.normal(0, 0.35, n_snps)
    gwas_se = np.full(n_snps, 0.08)
    gwas = pd.DataFrame({
        "SNP": snps,
        "CHR": chrom,
        "BP": bp,
        "effect_allele": a1,
        "other_allele": a2,
        "beta": gwas_z * gwas_se,
        "se": gwas_se,
        "pval": two_sided_p_from_z(gwas_z),
    })
    gwas.to_csv(out_dir / "gwas.tsv", sep="\t", index=False)

    tissue_weights = {
        "retina": np.array([1.0, 0.25, 0.35]),
        "blood": np.array([0.25, 1.0, 0.20]),
        "artery": np.array([0.75, 0.30, 0.55]),
    }
    manifest_rows = [{
        "type": "gwas",
        "name": "synthetic_trait",
        "path": str(rel_out / "gwas.tsv").replace("\\", "/"),
    }]
    for tissue in TISSUES:
        z = latent * tissue_weights[tissue][axis] + rng.normal(0, 0.45, n_snps)
        eqtl_a1 = a1.copy()
        eqtl_a2 = a2.copy()
        eqtl_z = z.copy()
        flip_idx = np.arange(0, n_snps, 37)
        eqtl_a1[flip_idx] = a2[flip_idx]
        eqtl_a2[flip_idx] = a1[flip_idx]
        eqtl_z[flip_idx] *= -1.0
        mismatch_idx = np.arange(19, n_snps, 211)
        eqtl_a1[mismatch_idx] = "A"
        eqtl_a2[mismatch_idx] = "T"

        eqtl_se = np.full(n_snps, 0.10)
        eqtl = pd.DataFrame({
            "SNP": snps,
            "A1": eqtl_a1,
            "A2": eqtl_a2,
            "BETA": eqtl_z * eqtl_se,
            "SE": eqtl_se,
            "CHR": chrom,
            "BP": bp,
            "TARGET_GENE": [f"MINI_GENE_{i:06d}" for i in range(1, n_snps + 1)],
            "P": two_sided_p_from_z(eqtl_z),
        })
        filename = f"eqtl_{tissue}.tsv"
        eqtl.to_csv(out_dir / filename, sep="\t", index=False)
        manifest_rows.append({
            "type": "eqtl",
            "name": tissue,
            "path": str(rel_out / filename).replace("\\", "/"),
        })

    blocks = []
    for c in sorted(set(chrom)):
        for start_bp in range(995_000, int(bp[chrom == c].max()) + 10_000, 100_000):
            blocks.append((f"chr{c}", start_bp, start_bp + 100_000))
    bed_header = pd.DataFrame(blocks, columns=["chr", "start", "stop"])
    bed_header.to_csv(out_dir / "ld_blocks_header.bed", sep="\t", index=False)
    bed_header.to_csv(out_dir / "ld_blocks_no_header.bed", sep="\t", index=False, header=False)

    manifest_rows.append({
        "type": "bed",
        "name": "synthetic_ld_blocks",
        "path": str(rel_out / "ld_blocks_header.bed").replace("\\", "/"),
    })
    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)

    readme = """# PRISMA 1000-SNP Mini-Fixture

This fixture is deterministic and fully synthetic. It is designed for software
validation only and is not biologically interpretable.

It includes one GWAS file, three eQTL tissue files, headered and no-header BED
LD-block files, several intentionally flipped alleles, and a small number of
allele mismatches for QC reporting.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Mini-fixture written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic PRISMA 1000-SNP mini-fixture.")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "mini_fixture_1000"))
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--n-snps", type=int, default=1000)
    args = parser.parse_args()
    generate_fixture(Path(args.out).resolve(), seed=args.seed, n_snps=args.n_snps)


if __name__ == "__main__":
    main()
