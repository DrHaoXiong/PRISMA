import pandas as pd

from qc import _qc_allele_harmonization


def test_allele_matching_counts_match_flip_and_mismatch():
    gwas = pd.DataFrame({
        "SNP": ["rs1", "rs2", "rs3"],
        "effect_allele": ["A", "C", "G"],
        "other_allele": ["C", "T", "A"],
    })
    eqtl = pd.DataFrame({
        "SNP": ["rs1", "rs2", "rs3"],
        "A1": ["A", "T", "T"],
        "A2": ["C", "C", "C"],
    })
    report = _qc_allele_harmonization(gwas, eqtl)
    assert report["matched_allele_count"] == 1
    assert report["flipped_allele_count"] == 1
    assert report["allele_mismatch_count"] == 1
