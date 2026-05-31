"""Run PRISMA across multiple seeds and summarize Factor B stability.

This utility is intentionally separate from the default PRISMA workflow because
multi-seed consensus runs can be slow on manuscript-scale inputs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _parse_seeds(seed_text: str) -> list[int]:
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def _load_factor_b(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    rank_cols = [c for c in df.columns if c.startswith("Rank_")]
    if not rank_cols:
        raise ValueError(f"No Rank_* columns found in {path}")
    return df[rank_cols]


def _align_similarity(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, object]:
    common_tissues = reference.index.intersection(candidate.index)
    if len(common_tissues) == 0:
        raise ValueError("No shared tissues between Factor B matrices.")
    ref = reference.loc[common_tissues].to_numpy(dtype=float)
    cand = candidate.loc[common_tissues].to_numpy(dtype=float)
    ref_norm = ref / (np.linalg.norm(ref, axis=0, keepdims=True) + 1e-12)
    cand_norm = cand / (np.linalg.norm(cand, axis=0, keepdims=True) + 1e-12)
    sim = ref_norm.T @ cand_norm
    row_ind, col_ind = linear_sum_assignment(-sim)
    aligned = sim[row_ind, col_ind]
    return {
        "assignment": {f"reference_rank_{int(r)}": f"candidate_rank_{int(c)}" for r, c in zip(row_ind, col_ind)},
        "mean_aligned_cosine": float(np.mean(aligned)),
        "min_aligned_cosine": float(np.min(aligned)),
        "aligned_cosines": [float(x) for x in aligned],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PRISMA multi-seed stability diagnostics.")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV.")
    parser.add_argument("--out", required=True, help="Output directory for seed-specific runs and summaries.")
    parser.add_argument("--rank", default="3", help="Rank passed to run_prisma.py.")
    parser.add_argument("--seeds", default="1,2,3,4,5", help="Comma-separated solver seeds.")
    parser.add_argument("--iter", type=int, default=20, help="ALS iterations per seed.")
    parser.add_argument("--phenotype-name", default=None, help="Optional phenotype name.")
    parser.add_argument("--ld-reference-mode", choices=["plink", "identity", "auto"], default="auto")
    parser.add_argument("--bfile", default=None, help="Optional PLINK reference prefix.")
    parser.add_argument("--allow-identity-ld", action="store_true")
    parser.add_argument("--ld-min-overlap", type=int, default=2)
    parser.add_argument("--ld-coverage-warning", type=float, default=0.80)
    parser.add_argument("--ld-coverage-fail", type=float, default=0.50)
    parser.add_argument("--allow-low-coverage", action="store_true")
    parser.add_argument("--block-assignment-warning", type=float, default=0.99)
    parser.add_argument("--block-assignment-fail", type=float, default=0.95)
    parser.add_argument("--allow-low-block-assignment", action="store_true")
    parser.add_argument("--allele-match-warning", type=float, default=0.90)
    parser.add_argument("--allele-match-fail", type=float, default=0.70)
    parser.add_argument("--allow-low-allele-match", action="store_true")
    parser.add_argument("--tissue-nonzero-warning", type=float, default=0.01)
    parser.add_argument("--tissue-nonzero-fail", type=float, default=0.001)
    parser.add_argument("--allow-low-tissue-nonzero", action="store_true")
    parser.add_argument("--gene-pruning-mode", choices=["strongest", "none", "top-k"], default="strongest")
    parser.add_argument("--gene-pruning-top-k", type=int, default=1)
    parser.add_argument("--require-mygene-for-ensembl", action="store_true")
    parser.add_argument("--allow-over-rank", action="store_true")
    parser.add_argument("--quiet-blocks", action="store_true")
    parser.add_argument("--no-run", action="store_true", help="Summarize existing seed_* outputs without running PRISMA.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)
    repo_root = Path(__file__).resolve().parents[1]
    run_prisma = repo_root / "run_prisma.py"

    for seed in seeds:
        seed_out = out_dir / f"seed_{seed}"
        if args.no_run:
            continue
        cmd = [
            sys.executable,
            str(run_prisma),
            "--manifest",
            args.manifest,
            "--out",
            str(seed_out),
            "--rank",
            str(args.rank),
            "--iter",
            str(args.iter),
            "--seed",
            str(seed),
            "--no_banner",
        ]
        if args.phenotype_name:
            cmd += ["--phenotype-name", args.phenotype_name]
        if args.bfile:
            cmd += ["--bfile", args.bfile]
        cmd += ["--ld-reference-mode", args.ld_reference_mode]
        cmd += ["--ld-min-overlap", str(args.ld_min_overlap)]
        cmd += ["--ld-coverage-warning", str(args.ld_coverage_warning)]
        cmd += ["--ld-coverage-fail", str(args.ld_coverage_fail)]
        cmd += ["--block-assignment-warning", str(args.block_assignment_warning)]
        cmd += ["--block-assignment-fail", str(args.block_assignment_fail)]
        cmd += ["--allele-match-warning", str(args.allele_match_warning)]
        cmd += ["--allele-match-fail", str(args.allele_match_fail)]
        cmd += ["--tissue-nonzero-warning", str(args.tissue_nonzero_warning)]
        cmd += ["--tissue-nonzero-fail", str(args.tissue_nonzero_fail)]
        cmd += ["--gene-pruning-mode", args.gene_pruning_mode]
        cmd += ["--gene-pruning-top-k", str(args.gene_pruning_top_k)]
        for flag in [
            "allow_identity_ld",
            "allow_low_coverage",
            "allow_low_block_assignment",
            "allow_low_allele_match",
            "allow_low_tissue_nonzero",
            "require_mygene_for_ensembl",
            "allow_over_rank",
            "quiet_blocks",
        ]:
            if getattr(args, flag):
                cmd.append("--" + flag.replace("_", "-"))
        subprocess.run(cmd, cwd=repo_root, check=True)

    factor_bs = {seed: _load_factor_b(out_dir / f"seed_{seed}" / "Factor_B_Tissues.csv") for seed in seeds}
    reference_seed = seeds[0]
    reference = factor_bs[reference_seed]
    rows = []
    pair_rows = []
    for seed in seeds:
        result = _align_similarity(reference, factor_bs[seed])
        rows.append({
            "reference_seed": reference_seed,
            "candidate_seed": seed,
            "mean_aligned_cosine": result["mean_aligned_cosine"],
            "min_aligned_cosine": result["min_aligned_cosine"],
            "assignment_json": json.dumps(result["assignment"], sort_keys=True),
        })
    for seed_a, seed_b in itertools.combinations(seeds, 2):
        result = _align_similarity(factor_bs[seed_a], factor_bs[seed_b])
        pair_rows.append({
            "seed_a": seed_a,
            "seed_b": seed_b,
            "mean_aligned_cosine": result["mean_aligned_cosine"],
            "min_aligned_cosine": result["min_aligned_cosine"],
            "assignment_json": json.dumps(result["assignment"], sort_keys=True),
        })

    pd.DataFrame(rows).to_csv(out_dir / "seed_stability_vs_reference.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out_dir / "seed_stability_pairwise.csv", index=False)
    summary = {
        "seeds": seeds,
        "reference_seed": reference_seed,
        "run_configuration": {
            "rank": args.rank,
            "iter": args.iter,
            "ld_reference_mode": args.ld_reference_mode,
            "bfile": args.bfile,
            "ld_min_overlap": args.ld_min_overlap,
            "ld_coverage_warning": args.ld_coverage_warning,
            "ld_coverage_fail": args.ld_coverage_fail,
            "allow_low_coverage": args.allow_low_coverage,
            "block_assignment_warning": args.block_assignment_warning,
            "block_assignment_fail": args.block_assignment_fail,
            "allow_low_block_assignment": args.allow_low_block_assignment,
            "allele_match_warning": args.allele_match_warning,
            "allele_match_fail": args.allele_match_fail,
            "allow_low_allele_match": args.allow_low_allele_match,
            "tissue_nonzero_warning": args.tissue_nonzero_warning,
            "tissue_nonzero_fail": args.tissue_nonzero_fail,
            "allow_low_tissue_nonzero": args.allow_low_tissue_nonzero,
            "gene_pruning_mode": args.gene_pruning_mode,
            "gene_pruning_top_k": args.gene_pruning_top_k,
            "require_mygene_for_ensembl": args.require_mygene_for_ensembl,
            "allow_over_rank": args.allow_over_rank,
        },
        "min_pairwise_aligned_cosine": float(min(row["min_aligned_cosine"] for row in pair_rows)) if pair_rows else 1.0,
        "mean_pairwise_aligned_cosine": float(np.mean([row["mean_aligned_cosine"] for row in pair_rows])) if pair_rows else 1.0,
    }
    (out_dir / "seed_stability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
