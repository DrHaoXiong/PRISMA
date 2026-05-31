from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


GWAS_ALIASES: dict[str, tuple[str, ...]] = {
    "SNP": ("SNP", "snp", "rsid", "rs_id", "variant_id", "variant", "marker", "markername"),
    "CHR": ("CHR", "chr", "chrom", "chromosome"),
    "BP": ("BP", "bp", "pos", "position", "base_pair_location", "basepair"),
    "effect_allele": ("effect_allele", "a1", "ea", "effect_allele_gwas", "alt", "allele1"),
    "other_allele": ("other_allele", "a2", "nea", "non_effect_allele", "ref", "allele2"),
    "beta": ("beta", "b", "effect", "effect_size", "log_odds", "or_beta"),
    "se": ("se", "stderr", "standard_error", "beta_se"),
    "pval": ("pval", "p", "p_value", "pvalue", "pval_nominal"),
}


EQTL_ALIASES: dict[str, tuple[str, ...]] = {
    "SNP": ("SNP", "snp", "rsid", "rs_id", "variant_id", "variant", "marker", "markername"),
    "A1": ("A1", "a1", "effect_allele", "ea", "alt", "allele1"),
    "A2": ("A2", "a2", "other_allele", "non_effect_allele", "nea", "ref", "allele2"),
    "BETA": ("BETA", "beta", "b", "effect", "slope", "beta_alt"),
    "SE": ("SE", "se", "stderr", "standard_error", "slope_se"),
    "CHR": ("CHR", "chr", "chrom", "chromosome"),
    "BP": ("BP", "bp", "pos", "position", "variant_pos", "base_pair_location", "basepair"),
    "TARGET_GENE": ("TARGET_GENE", "target_gene", "gene", "gene_symbol", "symbol", "hgnc_symbol", "gene_id"),
}


def canonicalize_column_name(name: str) -> str:
    """Normalize a user-provided column name for alias matching."""
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def build_column_rename_map(
    columns: Iterable[str],
    aliases: dict[str, tuple[str, ...]],
    context: str,
) -> dict[str, str]:
    """Build a collision-checked rename map from observed columns to canonical names."""
    alias_to_canonical: dict[str, str] = {}
    for canonical, values in aliases.items():
        for value in values + (canonical,):
            alias_to_canonical[canonicalize_column_name(value)] = canonical

    candidates: dict[str, list[str]] = {}
    for column in columns:
        canonical = alias_to_canonical.get(canonicalize_column_name(column))
        if canonical is not None and column != canonical:
            candidates.setdefault(canonical, []).append(column)
        elif canonical is not None and column == canonical:
            candidates.setdefault(canonical, []).append(column)

    rename: dict[str, str] = {}
    for canonical, observed in candidates.items():
        unique_observed = list(dict.fromkeys(observed))
        if len(unique_observed) > 1:
            raise ValueError(
                f"{context} schema has ambiguous columns for canonical field "
                f"'{canonical}': {unique_observed}. Keep exactly one."
            )
        rename[unique_observed[0]] = canonical
    return rename


def normalize_pandas_columns(
    df: pd.DataFrame,
    aliases: dict[str, tuple[str, ...]],
    context: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a copy with canonical columns plus the observed-to-canonical map."""
    rename = build_column_rename_map(df.columns, aliases, context)
    return df.rename(columns=rename), rename


def detect_delimiter(path: str | Path) -> str:
    """Detect comma versus tab delimiters without reading the full file."""
    path = Path(path)
    opener = gzip.open if path.suffix.lower() in {".gz", ".bgz"} else open
    with opener(path, "rt", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                return "," if line.count(",") > line.count("\t") else "\t"
    return "\t"


def read_table_with_detected_delimiter(path: str | Path) -> pd.DataFrame:
    """Read a CSV/TSV-like table using a lightweight delimiter sniff."""
    delimiter = detect_delimiter(path)
    return pd.read_csv(path, sep=delimiter)
