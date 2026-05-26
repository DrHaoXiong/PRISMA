import pandas as pd
import pytest

from qc import initialize_qc_report


def test_bad_gwas_schema_fails_actionably(tmp_path):
    gwas = tmp_path / "gwas.tsv"
    eqtl = tmp_path / "eqtl.tsv"
    bed = tmp_path / "blocks.bed"
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"SNP": ["rs1"]}).to_csv(gwas, sep="\t", index=False)
    pd.DataFrame({
        "SNP": ["rs1"], "A1": ["A"], "A2": ["C"], "BETA": [0.1], "SE": [0.1],
        "CHR": [1], "BP": [100], "TARGET_GENE": ["GENE1"],
    }).to_csv(eqtl, sep="\t", index=False)
    bed.write_text("chr1\t0\t1000\n", encoding="utf-8")
    pd.DataFrame([
        {"type": "gwas", "name": "trait", "path": str(gwas)},
        {"type": "eqtl", "name": "blood", "path": str(eqtl)},
        {"type": "bed", "name": "blocks", "path": str(bed)},
    ]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="GWAS file is missing required columns"):
        initialize_qc_report(str(manifest), str(tmp_path / "out"))
