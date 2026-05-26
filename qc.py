from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from partition import normalize_chromosome


GWAS_REQUIRED_COLUMNS = ["SNP", "CHR", "BP", "effect_allele", "other_allele", "beta", "se", "pval"]
EQTL_REQUIRED_COLUMNS = ["SNP", "A1", "A2", "BETA", "SE", "CHR", "BP", "TARGET_GENE"]
MANIFEST_REQUIRED_COLUMNS = ["type", "name", "path"]
VALID_ALLELES = {"A", "C", "G", "T"}
STRAND_AMBIGUOUS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}


def resolve_input_path(path_value: str, manifest_path: str | os.PathLike[str]) -> str:
    """Resolve manifest paths without changing existing relative-path behavior."""
    path = Path(str(path_value))
    if path.exists():
        return str(path)
    manifest_relative = Path(manifest_path).resolve().parent / path
    if manifest_relative.exists():
        return str(manifest_relative)
    return str(path)


def _read_table(path: str) -> pd.DataFrame:
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def _missing_counts(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    return {col: int(df[col].isna().sum()) if col in df.columns else int(len(df)) for col in columns}


def _allele_invalid_count(series: pd.Series) -> int:
    alleles = series.astype(str).str.upper()
    return int((~alleles.isin(VALID_ALLELES)).sum())


def _chrom_style(values: pd.Series) -> str:
    text = values.dropna().astype(str)
    if text.empty:
        return "unknown"
    has_chr = text.str.lower().str.startswith("chr").mean()
    if has_chr > 0.8:
        return "chr-prefixed"
    if has_chr < 0.2:
        return "numeric-or-unprefixed"
    return "mixed"


def _schema_failure(message: str, report: dict[str, Any]) -> None:
    report.setdefault("failures", []).append(message)


def initialize_qc_report(
    manifest_path: str,
    out_dir: str,
    allele_match_warning: float = 0.90,
    allele_match_fail: float = 0.70,
    allow_low_allele_match: bool = False,
) -> dict[str, Any]:
    """Run manifest, schema, and raw allele-overlap QC before loading tensors."""
    report: dict[str, Any] = {
        "manifest_path": manifest_path,
        "warnings": [],
        "failures": [],
        "manifest": {},
        "gwas": {},
        "eqtl": {},
        "allele_harmonization": {},
    }

    manifest = pd.read_csv(manifest_path)
    missing_manifest_cols = [c for c in MANIFEST_REQUIRED_COLUMNS if c not in manifest.columns]
    if missing_manifest_cols:
        _schema_failure(f"Manifest is missing required columns: {missing_manifest_cols}", report)
        write_qc_reports(report, out_dir)
        raise ValueError(report["failures"][-1])

    manifest = manifest.copy()
    manifest["type"] = manifest["type"].astype(str).str.lower()
    manifest["resolved_path"] = [
        resolve_input_path(path_value, manifest_path) for path_value in manifest["path"]
    ]

    duplicated_names = int(manifest["name"].duplicated().sum())
    missing_files = [
        str(row.path) for row in manifest.itertuples(index=False)
        if not os.path.exists(str(row.resolved_path))
    ]
    report["manifest"] = {
        "n_rows": int(len(manifest)),
        "row_types": sorted(manifest["type"].unique().tolist()),
        "duplicated_manifest_names": duplicated_names,
        "missing_files": missing_files,
    }

    if duplicated_names:
        report["warnings"].append(f"Manifest contains {duplicated_names} duplicated names.")
    if missing_files:
        _schema_failure(f"Manifest references missing files: {missing_files}", report)
    for required_type in ["gwas", "bed"]:
        if required_type not in set(manifest["type"]):
            _schema_failure(f"Manifest must contain a row with type='{required_type}'.", report)
    if "eqtl" not in set(manifest["type"]):
        _schema_failure("Manifest must contain at least one row with type='eqtl'.", report)

    if report["failures"]:
        write_qc_reports(report, out_dir)
        raise ValueError("; ".join(report["failures"]))

    gwas_row = manifest[manifest["type"] == "gwas"].iloc[0]
    gwas = _read_table(gwas_row["resolved_path"])
    report["gwas"] = _qc_gwas_schema(gwas)
    if report["gwas"]["missing_required_columns"]:
        _schema_failure(f"GWAS file is missing required columns: {report['gwas']['missing_required_columns']}", report)
        write_qc_reports(report, out_dir)
        raise ValueError(report["failures"][-1])

    eqtl_reports: dict[str, Any] = {}
    allele_reports: dict[str, Any] = {}
    for _, row in manifest[manifest["type"] == "eqtl"].iterrows():
        tissue = str(row["name"])
        eqtl = _read_table(row["resolved_path"])
        eqtl_reports[tissue] = _qc_eqtl_schema(eqtl)
        if eqtl_reports[tissue]["missing_required_columns"]:
            _schema_failure(
                f"eQTL file for {tissue} is missing required columns: "
                f"{eqtl_reports[tissue]['missing_required_columns']}",
                report,
            )
            continue
        allele_report = _qc_allele_harmonization(gwas, eqtl)
        allele_reports[tissue] = allele_report
        if allele_report["allele_match_rate"] < allele_match_fail and not allow_low_allele_match:
            _schema_failure(
                f"Allele match rate for {tissue} is {allele_report['allele_match_rate']:.3f}, "
                f"below fail threshold {allele_match_fail:.3f}.",
                report,
            )
        elif allele_report["allele_match_rate"] < allele_match_warning:
            report["warnings"].append(
                f"Allele match rate for {tissue} is {allele_report['allele_match_rate']:.3f}, "
                f"below warning threshold {allele_match_warning:.3f}."
            )

    report["eqtl"] = eqtl_reports
    report["allele_harmonization"] = allele_reports
    write_qc_reports(report, out_dir)
    if report["failures"]:
        raise ValueError("; ".join(report["failures"]))
    return report


def _qc_gwas_schema(gwas: pd.DataFrame) -> dict[str, Any]:
    missing_cols = [c for c in GWAS_REQUIRED_COLUMNS if c not in gwas.columns]
    report: dict[str, Any] = {
        "required_columns": GWAS_REQUIRED_COLUMNS,
        "missing_required_columns": missing_cols,
        "n_rows_raw": int(len(gwas)),
    }
    if missing_cols:
        return report
    beta = pd.to_numeric(gwas["beta"], errors="coerce")
    se = pd.to_numeric(gwas["se"], errors="coerce")
    pval = pd.to_numeric(gwas["pval"], errors="coerce")
    report.update({
        "n_unique_snps": int(gwas["SNP"].nunique()),
        "duplicated_snp_count": int(gwas["SNP"].duplicated().sum()),
        "missing_counts": _missing_counts(gwas, ["SNP", "CHR", "BP"]),
        "invalid_beta_se_pval_count": int(beta.isna().sum() + se.isna().sum() + pval.isna().sum()),
        "se_le_0_count": int((se <= 0).sum()),
        "allele_invalid_count": int(
            _allele_invalid_count(gwas["effect_allele"]) + _allele_invalid_count(gwas["other_allele"])
        ),
        "coordinate_missing_rate": float(gwas[["CHR", "BP"]].isna().any(axis=1).mean()),
        "chromosome_style": _chrom_style(gwas["CHR"]),
    })
    return report


def _qc_eqtl_schema(eqtl: pd.DataFrame) -> dict[str, Any]:
    missing_cols = [c for c in EQTL_REQUIRED_COLUMNS if c not in eqtl.columns]
    report: dict[str, Any] = {
        "required_columns": EQTL_REQUIRED_COLUMNS,
        "missing_required_columns": missing_cols,
        "n_rows_raw": int(len(eqtl)),
    }
    if missing_cols:
        return report
    se = pd.to_numeric(eqtl["SE"], errors="coerce")
    beta = pd.to_numeric(eqtl["BETA"], errors="coerce")
    report.update({
        "n_unique_snps": int(eqtl["SNP"].nunique()),
        "duplicated_snp_count": int(eqtl["SNP"].duplicated().sum()),
        "missing_required_field_counts": _missing_counts(eqtl, EQTL_REQUIRED_COLUMNS),
        "invalid_beta_se_count": int(beta.isna().sum() + se.isna().sum()),
        "se_le_0_count": int((se <= 0).sum()),
        "target_gene_missing_count": int(eqtl["TARGET_GENE"].isna().sum()),
        "nonzero_usable_row_count": int(((se > 0) & beta.notna() & eqtl["TARGET_GENE"].notna()).sum()),
        "chromosome_style": _chrom_style(eqtl["CHR"]),
    })
    return report


def _qc_allele_harmonization(gwas: pd.DataFrame, eqtl: pd.DataFrame) -> dict[str, Any]:
    g = gwas[["SNP", "effect_allele", "other_allele"]].drop_duplicates("SNP").copy()
    e = eqtl[["SNP", "A1", "A2"]].drop_duplicates("SNP").copy()
    merged = g.merge(e, on="SNP", how="inner")
    for col in ["effect_allele", "other_allele", "A1", "A2"]:
        merged[col] = merged[col].astype(str).str.upper()
    match = (merged["effect_allele"] == merged["A1"]) & (merged["other_allele"] == merged["A2"])
    flip = (merged["effect_allele"] == merged["A2"]) & (merged["other_allele"] == merged["A1"])
    ambiguous = [
        (a1, a2) in STRAND_AMBIGUOUS
        for a1, a2 in zip(merged["effect_allele"], merged["other_allele"])
    ]
    n_overlap = int(len(merged))
    n_match = int(match.sum())
    n_flip = int(flip.sum())
    n_ambiguous = int(np.sum(ambiguous))
    n_mismatch = int(n_overlap - n_match - n_flip)
    return {
        "n_gwas_eqtl_overlapping_snps": n_overlap,
        "matched_allele_count": n_match,
        "flipped_allele_count": n_flip,
        "strand_ambiguous_removed_count": n_ambiguous,
        "allele_mismatch_count": n_mismatch,
        "allele_match_rate": float((n_match + n_flip) / n_overlap) if n_overlap else 0.0,
    }


def add_tensor_and_ld_qc(
    report: dict[str, Any],
    aligned_df: pd.DataFrame,
    tissue_cols: list[str],
    block_defs: pd.DataFrame,
    partitioner,
    builder,
    out_dir: str,
    tissue_nonzero_warning: float = 0.01,
    tissue_nonzero_fail: float = 0.001,
    allow_low_tissue_nonzero: bool = False,
    ld_coverage_warning: float = 0.80,
    ld_coverage_fail: float = 0.50,
    allow_low_coverage: bool = False,
) -> dict[str, Any]:
    tensor_qc = _qc_tensor_coverage(aligned_df, tissue_cols)
    report["tensor_coverage"] = tensor_qc
    for tissue, rate in tensor_qc["nonzero_rate_per_tissue"].items():
        if rate < tissue_nonzero_fail and not allow_low_tissue_nonzero:
            report.setdefault("failures", []).append(
                f"Tissue nonzero rate for {tissue} is {rate:.5f}, below fail threshold {tissue_nonzero_fail:.5f}."
            )
        elif rate < tissue_nonzero_warning:
            report.setdefault("warnings", []).append(
                f"Tissue nonzero rate for {tissue} is {rate:.5f}, below warning threshold {tissue_nonzero_warning:.5f}."
            )

    ld_qc = _qc_ld_block_coverage(aligned_df, block_defs, partitioner, builder)
    report["ld_block_coverage"] = ld_qc
    fraction_empirical = ld_qc.get("fraction_snps_with_empirical_ld", 0.0)
    if builder.bfile_path is not None and fraction_empirical < ld_coverage_fail and not allow_low_coverage:
        report.setdefault("failures", []).append(
            f"Fraction of SNPs with empirical LD is {fraction_empirical:.3f}, below fail threshold {ld_coverage_fail:.3f}."
        )
    elif builder.bfile_path is not None and fraction_empirical < ld_coverage_warning:
        report.setdefault("warnings", []).append(
            f"Fraction of SNPs with empirical LD is {fraction_empirical:.3f}, below warning threshold {ld_coverage_warning:.3f}."
        )
    if ld_qc.get("fraction_snps_assigned_to_block", 0.0) < 0.50:
        report.setdefault("warnings", []).append(
            "Fewer than 50% of tensor SNPs were assigned to LD blocks; check genome build and chromosome naming."
        )
    write_qc_reports(report, out_dir)
    if report.get("failures"):
        raise ValueError("; ".join(report["failures"]))
    return report


def _qc_tensor_coverage(aligned_df: pd.DataFrame, tissue_cols: list[str]) -> dict[str, Any]:
    nonzero_entries = {}
    nonzero_rates = {}
    for col in tissue_cols:
        values = aligned_df[col].fillna(0.0).to_numpy()
        nonzero_entries[col.replace("_Z", "")] = int(np.count_nonzero(values))
        nonzero_rates[col.replace("_Z", "")] = float(np.count_nonzero(values) / max(len(values), 1))
    tissue_matrix = aligned_df[tissue_cols].fillna(0.0).to_numpy() if tissue_cols else np.zeros((len(aligned_df), 0))
    return {
        "n_tensor_snps": int(aligned_df["SNP"].nunique()) if "SNP" in aligned_df.columns else int(len(aligned_df)),
        "n_tensor_genes": int(
            aligned_df[[c for c in aligned_df.columns if c.endswith("_GENE")]].replace("no_eqtl", np.nan).nunique().sum()
        ) if any(c.endswith("_GENE") for c in aligned_df.columns) else None,
        "n_tissues": int(len(tissue_cols)),
        "nonzero_entries_per_tissue": nonzero_entries,
        "nonzero_rate_per_tissue": nonzero_rates,
        "all_zero_snp_rows": int((np.abs(tissue_matrix).sum(axis=1) == 0).sum()) if tissue_cols else int(len(aligned_df)),
        "all_zero_tissue_columns": [k for k, v in nonzero_entries.items() if v == 0],
    }


def _qc_ld_block_coverage(aligned_df: pd.DataFrame, block_defs: pd.DataFrame, partitioner, builder) -> dict[str, Any]:
    assigned_snps: set[str] = set()
    n_empty = 0
    n_insufficient = 0
    n_empirical_blocks = 0
    n_identity_blocks = 0
    for _, row in block_defs.iterrows():
        chrom = normalize_chromosome(row["chr"])
        start = int(row["start"])
        stop = int(row["stop"])
        mask = (
            (partitioner.data_df["CHR"] == chrom)
            & (partitioner.data_df["BP"] >= start)
            & (partitioner.data_df["BP"] < stop)
        )
        sub_df = partitioner.data_df[mask]
        if len(sub_df) == 0:
            n_empty += 1
            continue
        assigned_snps.update(sub_df["SNP"].astype(str).tolist())
        overlap = builder.get_reference_overlap_count(sub_df["SNP"].astype(str).tolist())
        if builder.bfile_path is not None and overlap >= builder.ld_min_overlap:
            n_empirical_blocks += 1
        elif builder.bfile_path is not None:
            n_insufficient += 1
            n_identity_blocks += 1
        else:
            n_identity_blocks += 1
    n_tensor_snps = int(aligned_df["SNP"].nunique())
    n_assigned = len(assigned_snps)
    if builder.bfile_path is not None:
        n_with_ref = sum(1 for snp in aligned_df["SNP"].astype(str) if builder.get_reference_overlap_count([snp]) > 0)
    else:
        n_with_ref = 0
    return {
        "n_ld_blocks_loaded": int(partitioner.block_stats.get("n_blocks_loaded", len(block_defs))),
        "n_ld_blocks_used": int(len(block_defs)),
        "n_blocks_dropped_empty": int(n_empty),
        "n_blocks_dropped_insufficient_snps": int(n_insufficient),
        "n_blocks_identity_laplacian": int(n_identity_blocks),
        "n_blocks_empirical_laplacian": int(n_empirical_blocks),
        "n_snps_assigned_to_block": int(n_assigned),
        "n_snps_unassigned_to_block": int(n_tensor_snps - n_assigned),
        "fraction_snps_assigned_to_block": float(n_assigned / max(n_tensor_snps, 1)),
        "fraction_snps_with_empirical_ld": float(n_with_ref / max(n_tensor_snps, 1)),
        "ld_mode": builder.ld_reference_mode,
        "bfile_path": builder.bfile_path,
        "chromosome_style_bed": _chrom_style(block_defs["chr"]),
        "chromosome_style_tensor": _chrom_style(aligned_df["CHR"]),
    }


def write_qc_reports(report: dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "qc_report.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    rows = []
    _flatten_report(report, rows)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(out_dir, "qc_summary.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "qc_report.tsv"), sep="\t", index=False)

    with open(os.path.join(out_dir, "qc_report.txt"), "w", encoding="utf-8") as handle:
        handle.write("PRISMA QC report\n")
        handle.write("================\n")
        for warning in report.get("warnings", []):
            handle.write(f"WARNING: {warning}\n")
        for failure in report.get("failures", []):
            handle.write(f"FAILURE: {failure}\n")
        handle.write("\nKey metrics\n")
        for row in rows:
            handle.write(f"{row['section']}.{row['metric']}: {row['value']}\n")


def _flatten_report(obj: Any, rows: list[dict[str, str]], prefix: str = "") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten_report(value, rows, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(obj, list):
        rows.append({"section": prefix.rsplit(".", 1)[0] if "." in prefix else prefix, "metric": prefix.rsplit(".", 1)[-1], "value": "; ".join(map(str, obj))})
    else:
        rows.append({"section": prefix.rsplit(".", 1)[0] if "." in prefix else prefix, "metric": prefix.rsplit(".", 1)[-1], "value": str(obj)})


def print_qc_summary(report: dict[str, Any]) -> None:
    print("\n[QC] Summary")
    print(f"   Warnings: {len(report.get('warnings', []))}")
    print(f"   Failures: {len(report.get('failures', []))}")
    tensor = report.get("tensor_coverage", {})
    if tensor:
        print(f"   Tensor SNPs: {tensor.get('n_tensor_snps')}; tissues: {tensor.get('n_tissues')}")
    ld = report.get("ld_block_coverage", {})
    if ld:
        print(
            "   LD blocks used: "
            f"{ld.get('n_ld_blocks_used')}; SNP block assignment: "
            f"{ld.get('fraction_snps_assigned_to_block', 0):.3f}; empirical LD SNP coverage: "
            f"{ld.get('fraction_snps_with_empirical_ld', 0):.3f}"
        )
