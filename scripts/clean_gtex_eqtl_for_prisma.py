#!/usr/bin/env python3
"""
Clean raw GTEx eQTL association tables for PRISMA.

Raw GTEx eQTL association files can contain multiple gene-level associations
for the same rsID. PRISMA uses a SNP x tissue input matrix, so this utility
retains one representative association per SNP: the row with the largest
absolute eQTL Z-score, where Z = beta / se.

Expected raw GTEx columns include:
    pvalue, gene_id, molecular_trait_id, beta, se, chromosome, position,
    ref, alt, rsid

Output schema:
    SNP, A1, A2, BETA, SE, CHR, BP, TARGET_GENE, P

Allele convention:
    GTEx beta is interpreted with respect to ALT, so A1=ALT and A2=REF.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "pvalue",
    "gene_id",
    "molecular_trait_id",
    "beta",
    "se",
    "chromosome",
    "position",
    "ref",
    "alt",
    "rsid",
]


def load_epi_symbol_map(epi_path: Optional[Path]) -> Dict[str, str]:
    """Load Ensembl gene ID to gene symbol mapping from an optional SMR .epi file."""
    if epi_path is None:
        return {}
    if not epi_path.exists():
        raise FileNotFoundError(f"SMR .epi mapping file not found: {epi_path}")

    epi = pd.read_csv(
        epi_path,
        sep=r"\s+",
        header=None,
        usecols=[1, 4],
        names=["gene_id", "symbol"],
        dtype=str,
        engine="python",
    )
    epi["gene_id"] = epi["gene_id"].str.split(".").str[0]
    epi = epi.dropna(subset=["gene_id", "symbol"]).drop_duplicates("gene_id")
    return dict(zip(epi["gene_id"], epi["symbol"]))


def iter_chunk_best(
    input_path: Path,
    chunksize: int,
    max_rows: Optional[int] = None,
) -> Iterable[pd.DataFrame]:
    """Yield the best representative eQTL row per rsID within each chunk."""
    rows_seen = 0
    reader = pd.read_csv(
        input_path,
        sep="\t",
        compression="infer",
        usecols=RAW_COLUMNS,
        chunksize=chunksize,
        dtype={
            "pvalue": "float64",
            "gene_id": "string",
            "molecular_trait_id": "string",
            "beta": "float64",
            "se": "float64",
            "chromosome": "string",
            "position": "Int64",
            "ref": "string",
            "alt": "string",
            "rsid": "string",
        },
    )

    for chunk_idx, chunk in enumerate(reader, start=1):
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()

        rows_seen += len(chunk)
        chunk = chunk.rename(
            columns={
                "rsid": "SNP",
                "alt": "A1",
                "ref": "A2",
                "beta": "BETA",
                "se": "SE",
                "chromosome": "CHR",
                "position": "BP",
                "pvalue": "P",
            }
        )

        chunk["SNP"] = chunk["SNP"].astype("string")
        chunk = chunk[
            chunk["SNP"].notna()
            & (chunk["SNP"] != "")
            & (chunk["SNP"] != "NA")
            & chunk["BETA"].notna()
            & chunk["SE"].notna()
            & (chunk["SE"] > 0)
            & chunk["P"].notna()
        ].copy()

        chunk["CHR"] = pd.to_numeric(chunk["CHR"], errors="coerce")
        chunk["BP"] = pd.to_numeric(chunk["BP"], errors="coerce")
        chunk = chunk[chunk["CHR"].between(1, 22) & chunk["BP"].notna()].copy()
        if len(chunk) == 0:
            print(f"Chunk {chunk_idx}: no valid autosomal rsID rows retained; skipping")
            continue

        chunk["CHR"] = chunk["CHR"].astype(int)
        chunk["BP"] = chunk["BP"].astype(int)
        chunk["Z"] = chunk["BETA"] / chunk["SE"]
        chunk["Abs_Z"] = chunk["Z"].abs()
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=["Abs_Z"])
        if len(chunk) == 0:
            print(f"Chunk {chunk_idx}: no finite beta/se rows retained; skipping")
            continue

        idx = chunk.groupby("SNP", sort=False)["Abs_Z"].idxmax()
        best = chunk.loc[idx].copy()
        best["Source_Chunk"] = chunk_idx
        yield best


def reduce_best(current: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    """Combine best-per-SNP chunks and keep the largest absolute Z per SNP."""
    if current is None or len(current) == 0:
        return new
    combined = pd.concat([current, new], ignore_index=True)
    idx = combined.groupby("SNP", sort=False)["Abs_Z"].idxmax()
    return combined.loc[idx].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw GTEx eQTL associations to PRISMA-formatted eQTL input."
    )
    parser.add_argument("--input", required=True, help="Raw GTEx eQTL table, for example Whole_Blood.tsv.gz")
    parser.add_argument("--output", required=True, help="Output PRISMA-formatted eQTL table")
    parser.add_argument(
        "--epi-map",
        default=None,
        help="Optional SMR .epi file used only to map Ensembl IDs to gene symbols",
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--summary-output", default=None, help="Optional CSV summary path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output) if args.summary_output else None
    epi_path = Path(args.epi_map) if args.epi_map else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    symbol_map = load_epi_symbol_map(epi_path)
    print(f"Loaded {len(symbol_map):,} gene-symbol mappings")

    best_all: Optional[pd.DataFrame] = None
    for best_chunk in iter_chunk_best(input_path, args.chunksize, args.max_rows):
        best_all = reduce_best(best_all, best_chunk)
        print(
            f"Processed chunk {int(best_chunk['Source_Chunk'].iloc[0])}: "
            f"chunk_best={len(best_chunk):,}, retained_unique_snps={len(best_all):,}"
        )

    if best_all is None or len(best_all) == 0:
        raise RuntimeError("No valid GTEx eQTL rows were retained.")

    best_all["gene_id_clean"] = (
        best_all["gene_id"]
        .fillna(best_all["molecular_trait_id"])
        .astype(str)
        .str.split(".")
        .str[0]
    )
    best_all["TARGET_GENE"] = best_all["gene_id_clean"].map(symbol_map).fillna(best_all["gene_id_clean"])

    final_cols = ["SNP", "A1", "A2", "BETA", "SE", "CHR", "BP", "TARGET_GENE", "P"]
    final = best_all[final_cols].sort_values("SNP").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, sep="\t", index=False)

    summary = {
        "Input_File": str(input_path),
        "Output_File": str(output_path),
        "Chunksize": args.chunksize,
        "Max_Rows": args.max_rows if args.max_rows is not None else "all",
        "N_Output_Unique_SNPs": len(final),
        "N_Unique_Target_Genes": final["TARGET_GENE"].nunique(),
        "N_Target_Genes_Mapped_To_Symbol": int(final["TARGET_GENE"].ne(best_all["gene_id_clean"]).sum()),
        "Representative_Rule": "per SNP, retain largest absolute eQTL Z-score beta/se",
        "Allele_Convention": "A1=ALT/effect allele, A2=REF",
    }

    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("Done.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
