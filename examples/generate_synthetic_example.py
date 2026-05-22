#!/usr/bin/env python3
"""
Generate a tiny PRISMA-compatible synthetic dataset.

The files produced by this script are intentionally small and fully synthetic.
They are meant for software testing and tutorial use only, not for biological
interpretation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


TISSUES = ["Retina", "Artery", "Blood", "Brain", "Pancreas"]


def two_sided_p_from_z(z: np.ndarray) -> np.ndarray:
    """Compute two-sided normal p values without requiring scipy."""
    return np.array([math.erfc(abs(float(x)) / math.sqrt(2.0)) for x in z])


def choose_alleles(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    allele_pairs = np.array([
        ["A", "C"],
        ["A", "G"],
        ["C", "A"],
        ["C", "T"],
        ["G", "A"],
        ["G", "T"],
        ["T", "C"],
        ["T", "G"],
    ])
    idx = rng.integers(0, len(allele_pairs), size=n)
    selected = allele_pairs[idx]
    return selected[:, 0], selected[:, 1]


def generate_synthetic_example(out_dir: Path, seed: int = 20260515, n_snps: int = 240) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = out_dir.parents[1]
    rel_out = out_dir.relative_to(repo_root)

    snps = np.array([f"rsPRISMA{i:05d}" for i in range(1, n_snps + 1)])
    chrom = np.repeat(np.arange(1, 4), math.ceil(n_snps / 3))[:n_snps]
    bp_offsets = np.tile(np.arange(1, math.ceil(n_snps / 3) + 1), 3)[:n_snps]
    bp = 1_000_000 + bp_offsets * 10_000
    a1, a2 = choose_alleles(rng, n_snps)

    axis = np.arange(n_snps) % 3
    latent = rng.normal(loc=0.0, scale=1.0, size=n_snps)
    latent += np.choose(axis, [1.2, 1.0, 0.9]) * rng.choice([-1.0, 1.0], size=n_snps)

    # Rows are tissues, columns are three synthetic disease axes.
    tissue_axis_weights = np.array([
        [1.00, 0.35, 0.45],  # Retina
        [0.85, 0.20, 0.10],  # Artery
        [0.30, 1.00, 0.15],  # Blood
        [0.10, 0.15, 1.00],  # Brain
        [0.35, 0.20, 0.55],  # Pancreas
    ])

    gwas_axis_weights = np.array([1.00, 0.85, 0.75])
    gwas_z = latent * gwas_axis_weights[axis] + rng.normal(0.0, 0.65, size=n_snps)
    gwas_se = np.full(n_snps, 0.08)
    gwas_beta = gwas_z * gwas_se

    gwas = pd.DataFrame({
        "SNP": snps,
        "CHR": chrom,
        "BP": bp,
        "effect_allele": a1,
        "other_allele": a2,
        "beta": gwas_beta,
        "se": gwas_se,
        "pval": two_sided_p_from_z(gwas_z),
    })
    gwas.to_csv(out_dir / "synthetic_dr_gwas.tsv", sep="\t", index=False)

    eqtl_paths = []
    for tissue_idx, tissue in enumerate(TISSUES):
        tissue_weight = tissue_axis_weights[tissue_idx, axis]
        eqtl_z = latent * tissue_weight + rng.normal(0.0, 0.55, size=n_snps)

        eqtl_a1 = a1.copy()
        eqtl_a2 = a2.copy()
        eqtl_z_for_file = eqtl_z.copy()

        flip_mask = rng.random(n_snps) < 0.08
        eqtl_a1[flip_mask] = a2[flip_mask]
        eqtl_a2[flip_mask] = a1[flip_mask]
        eqtl_z_for_file[flip_mask] *= -1.0

        eqtl_se = np.full(n_snps, 0.10)
        eqtl = pd.DataFrame({
            "SNP": snps,
            "A1": eqtl_a1,
            "A2": eqtl_a2,
            "BETA": eqtl_z_for_file * eqtl_se,
            "SE": eqtl_se,
            "TARGET_GENE": [f"GENE_DEMO_{i:05d}" for i in range(1, n_snps + 1)],
        })
        filename = f"synthetic_{tissue.lower()}_eqtl.tsv"
        eqtl.to_csv(out_dir / filename, sep="\t", index=False)
        eqtl_paths.append((tissue, rel_out / filename))

    blocks = []
    for c in sorted(set(chrom)):
        chrom_bp = bp[chrom == c]
        blocks.append((c, int(chrom_bp.min() - 5_000), int(chrom_bp.min() + 300_000)))
        blocks.append((c, int(chrom_bp.min() + 300_000), int(chrom_bp.max() + 20_000)))

    bed = pd.DataFrame(blocks, columns=["chr", "start", "stop"])
    bed.to_csv(out_dir / "synthetic_ld_blocks.bed", sep="\t", index=False)

    manifest_rows = [{
        "type": "gwas",
        "name": "Synthetic_DR",
        "path": str(rel_out / "synthetic_dr_gwas.tsv").replace("\\", "/"),
    }]
    for tissue, path in eqtl_paths:
        manifest_rows.append({
            "type": "eqtl",
            "name": tissue,
            "path": str(path).replace("\\", "/"),
        })
    manifest_rows.append({
        "type": "bed",
        "name": "Synthetic_LD_Blocks",
        "path": str(rel_out / "synthetic_ld_blocks.bed").replace("\\", "/"),
    })

    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)

    print(f"Synthetic PRISMA example written to: {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.csv'}")
    print(f"SNPs: {n_snps}; tissues: {len(TISSUES)}; LD blocks: {len(blocks)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic PRISMA example dataset.")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "synthetic_data"),
        help="Output directory for generated synthetic files.",
    )
    parser.add_argument("--seed", type=int, default=20260515, help="Random seed.")
    parser.add_argument("--n_snps", type=int, default=240, help="Number of synthetic SNPs.")
    args = parser.parse_args()

    generate_synthetic_example(Path(args.out).resolve(), seed=args.seed, n_snps=args.n_snps)


if __name__ == "__main__":
    main()
