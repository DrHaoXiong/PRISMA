import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_small_fixture(tmp_path: Path, lowercase_eqtl: bool = False, duplicate_genes: bool = False) -> Path:
    snps = [f"rsT{i}" for i in range(1, 9)]
    chrom = [1] * len(snps)
    bp = [1000 + i * 1000 for i in range(len(snps))]
    a1 = ["A", "C", "G", "T", "A", "C", "G", "T"]
    a2 = ["C", "T", "A", "G", "G", "A", "T", "C"]
    gwas = pd.DataFrame({
        "snp": snps,
        "chr": chrom,
        "bp": bp,
        "a1": a1,
        "a2": a2,
        "beta": [0.20, 0.12, -0.11, 0.18, 0.15, -0.09, 0.10, 0.14],
        "se": [0.05] * len(snps),
        "p": [1e-4, 0.01, 0.02, 5e-4, 0.003, 0.04, 0.03, 0.006],
    })
    gwas_path = tmp_path / "gwas.tsv"
    gwas.to_csv(gwas_path, sep="\t", index=False)

    genes = [f"GENE{i}" for i in range(1, 9)]
    if duplicate_genes:
        genes = ["GENE1", "GENE1", "GENE2", "GENE2", "GENE3", "GENE3", "GENE4", "GENE4"]

    eqtl_base = pd.DataFrame({
        "SNP": snps,
        "A1": a1,
        "A2": a2,
        "BETA": [0.30, 0.25, -0.22, 0.28, 0.26, -0.18, 0.19, 0.21],
        "SE": [0.05] * len(snps),
        "CHR": chrom,
        "BP": bp,
        "TARGET_GENE": genes,
    })
    manifest_rows = [{"type": "gwas", "name": "toy_trait", "path": str(gwas_path)}]
    for tissue in ["retina", "blood"]:
        eqtl = eqtl_base.copy()
        if tissue == "blood":
            eqtl["BETA"] = eqtl["BETA"] * 0.8
        if lowercase_eqtl:
            eqtl = eqtl.rename(columns={c: c.lower() for c in eqtl.columns})
        eqtl_path = tmp_path / f"eqtl_{tissue}.tsv"
        eqtl.to_csv(eqtl_path, sep="\t", index=False)
        manifest_rows.append({"type": "eqtl", "name": tissue, "path": str(eqtl_path)})

    bed_path = tmp_path / "blocks.bed"
    bed_path.write_text("chr1\t0\t20000\n", encoding="utf-8")
    manifest_rows.append({"type": "bed", "name": "blocks", "path": str(bed_path)})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    return manifest


def _run_prisma(repo: Path, manifest: Path, out: Path, extra_args: list[str] | None = None):
    cmd = [
        sys.executable,
        str(repo / "run_prisma.py"),
        "--manifest",
        str(manifest),
        "--out",
        str(out),
        "--rank",
        "1",
        "--iter",
        "1",
        "--ld-reference-mode",
        "identity",
        "--allow-identity-ld",
        "--allow-low-tissue-nonzero",
        "--no_banner",
        "--quiet-blocks",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=repo, text=True, capture_output=True)


def test_lowercase_and_alias_columns_are_normalized(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    manifest = _write_small_fixture(tmp_path, lowercase_eqtl=True)
    out = tmp_path / "out"
    result = _run_prisma(repo, manifest, out)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (out / "Factor_A_SNPs.csv").exists()
    qc = json.loads((out / "qc_report.json").read_text(encoding="utf-8"))
    assert qc["gwas"]["column_mapping"]["snp"] == "SNP"
    assert qc["eqtl"]["retina"]["column_mapping"]["target_gene"] == "TARGET_GENE"


def test_multiple_gwas_rows_fail_clearly(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    manifest = _write_small_fixture(tmp_path)
    df = pd.read_csv(manifest)
    df = pd.concat([df.iloc[[0]], df], ignore_index=True)
    df.to_csv(manifest, index=False)
    result = _run_prisma(repo, manifest, tmp_path / "out")
    assert result.returncode != 0
    assert "requires exactly one GWAS row" in (result.stdout + result.stderr)


def test_no_overlap_fails_actionably(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    manifest = _write_small_fixture(tmp_path)
    for path in tmp_path.glob("eqtl_*.tsv"):
        eqtl = pd.read_csv(path, sep="\t")
        eqtl["SNP"] = [f"rsNO{i}" for i in range(len(eqtl))]
        eqtl.to_csv(path, sep="\t", index=False)
    result = _run_prisma(repo, manifest, tmp_path / "out")
    assert result.returncode != 0
    assert "No overlapping SNPs" in (result.stdout + result.stderr)


def test_gene_pruning_modes_record_compression(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    manifest = _write_small_fixture(tmp_path, duplicate_genes=True)
    strongest_out = tmp_path / "strongest"
    none_out = tmp_path / "none"
    result = _run_prisma(repo, manifest, strongest_out)
    assert result.returncode == 0, result.stdout + result.stderr
    result = _run_prisma(repo, manifest, none_out, ["--gene-pruning-mode", "none"])
    assert result.returncode == 0, result.stdout + result.stderr
    strongest_rows = len(pd.read_csv(strongest_out / "Factor_A_SNPs.csv"))
    none_rows = len(pd.read_csv(none_out / "Factor_A_SNPs.csv"))
    assert strongest_rows == 4
    assert none_rows == 8
    qc = json.loads((strongest_out / "qc_report.json").read_text(encoding="utf-8"))
    assert qc["gene_pruning"]["mode"] == "strongest"
    assert qc["gene_pruning"]["compression_rate"] == 0.5


def test_over_rank_fails_without_explicit_permission(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    manifest = _write_small_fixture(tmp_path)
    result = _run_prisma(repo, manifest, tmp_path / "out", ["--rank", "4"])
    assert result.returncode != 0
    assert "exceeds the number of tissue columns" in (result.stdout + result.stderr)


def test_incomplete_ld_block_assignment_fails_by_default(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    manifest = _write_small_fixture(tmp_path)
    bed_path = tmp_path / "blocks.bed"
    bed_path.write_text("chr1\t3000\t20000\n", encoding="utf-8")
    result = _run_prisma(repo, manifest, tmp_path / "out")
    assert result.returncode != 0
    assert "assigned to LD blocks" in (result.stdout + result.stderr)


def test_incomplete_ld_block_assignment_can_be_explicitly_allowed(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    manifest = _write_small_fixture(tmp_path)
    bed_path = tmp_path / "blocks.bed"
    bed_path.write_text("chr1\t3000\t20000\n", encoding="utf-8")
    out = tmp_path / "out"
    result = _run_prisma(repo, manifest, out, ["--allow-low-block-assignment"])
    assert result.returncode == 0, result.stdout + result.stderr
    qc = json.loads((out / "qc_report.json").read_text(encoding="utf-8"))
    assert qc["ld_block_coverage"]["n_snps_unassigned_to_block"] == 2
    assert any("will not appear in Factor_A_SNPs.csv" in w for w in qc["warnings"])


def test_strand_ambiguous_snps_are_excluded_by_default(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    manifest = _write_small_fixture(tmp_path)
    gwas = pd.read_csv(tmp_path / "gwas.tsv", sep="\t")
    gwas.loc[0, ["a1", "a2"]] = ["A", "T"]
    gwas.to_csv(tmp_path / "gwas.tsv", sep="\t", index=False)
    for path in tmp_path.glob("eqtl_*.tsv"):
        eqtl = pd.read_csv(path, sep="\t")
        eqtl.loc[0, ["A1", "A2"]] = ["A", "T"]
        eqtl.to_csv(path, sep="\t", index=False)

    default_out = tmp_path / "default"
    keep_out = tmp_path / "keep"
    result = _run_prisma(repo, manifest, default_out)
    assert result.returncode == 0, result.stdout + result.stderr
    result = _run_prisma(repo, manifest, keep_out, ["--keep-strand-ambiguous"])
    assert result.returncode == 0, result.stdout + result.stderr

    assert len(pd.read_csv(default_out / "Factor_A_SNPs.csv")) == 7
    assert len(pd.read_csv(keep_out / "Factor_A_SNPs.csv")) == 8
    qc = json.loads((default_out / "qc_report.json").read_text(encoding="utf-8"))
    assert qc["allele_harmonization"]["retina"]["strand_ambiguous_removed_count"] == 1
