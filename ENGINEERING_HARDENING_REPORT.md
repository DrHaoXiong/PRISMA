# Engineering Hardening Report

## Files Modified

- `.gitignore`
- `README.md`
- `requirements.txt`
- `environment.yml`
- `builder.py`
- `loader.py`
- `partition.py`
- `qc.py`
- `run_prisma.py`
- `tune_rank.py`
- `scripts/check_environment.py`
- `examples/data_manifest_template.csv`
- `examples/generate_synthetic_example.py`
- `examples/generate_mini_fixture_1000.py`
- `examples/synthetic_data/README.md`
- `examples/synthetic_data/manifest.csv`
- `examples/mini_fixture_1000/*`
- `tests/*`

## Issues Fixed

| Issue | Status | Implementation summary | Tests added |
|---|---|---|---|
| Missing `pyarrow` dependency | fixed | Added `pyarrow`, `scikit-learn`, and `pytest` to `requirements.txt`; added `environment.yml`; added `scripts/check_environment.py`. | `tests/test_imports.py` |
| BED no-header parsing | fixed | Added header autodetection, comma/tab/whitespace delimiter support, chromosome normalization, numeric coordinate validation, and block-count reporting in `partition.py`. | `tests/test_partition_bed_header.py`, `tests/test_partition_bed_no_header.py`, `tests/test_partition_bed_whitespace.py` |
| CLI LD reference parameters | fixed | Added `--bfile`, `--ld-reference-mode`, `--allow-identity-ld`, `--ld-min-overlap`, LD coverage thresholds, and low-coverage override flags. `builder.py` now tracks empirical versus identity Laplacian usage. | `tests/test_builder_plink_ld.py`; end-to-end PLINK mini-run tested manually |
| Hard-coded phenotype name | fixed | Added `--phenotype-name`; default inferred from GWAS manifest name; `Factor_C_Phenotypes.csv` now uses the selected phenotype. | `tests/test_phenotype_name.py` |
| Hard QC report | fixed | Added `qc.py` with manifest, schema, allele harmonization, tensor coverage, LD block coverage, and coordinate consistency metrics. Runs write `qc_report.json`, `qc_summary.csv`, `qc_report.tsv`, and `qc_report.txt` before fitting. | `tests/test_qc_schema.py`, `tests/test_qc_allele_matching.py`, `tests/test_qc_ld_block_coverage.py` |
| Rank selection consistency | fixed | Centralized rank selection in `tune_rank.select_rank`; both `run_prisma.py --rank auto` and `tune_rank.py` call the same function and write `rank_diagnostics.csv` / `rank_selection.json`. | `tests/test_rank_selection_consistency.py` |
| Mini-fixture | fixed | Added deterministic synthetic 1000-SNP mini-fixture generator and generated fixture with GWAS, three eQTL tissues, headered/no-header BED, flipped alleles, and mismatch alleles. | `tests/test_cli_smoke_mini_fixture.py` |

## Commands Tested

Dependency installation and environment check:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/check_environment.py
```

Pytest:

```bash
.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
13 passed
```

Synthetic mini-fixture smoke test:

```bash
.venv\Scripts\python.exe examples/generate_mini_fixture_1000.py
.venv\Scripts\python.exe run_prisma.py ^
  --manifest examples/mini_fixture_1000/manifest.csv ^
  --out results/mini_fixture_test ^
  --rank auto ^
  --max-rank 5 ^
  --phenotype-name synthetic_trait ^
  --ld-reference-mode identity ^
  --allow-identity-ld ^
  --iter 5 ^
  --quiet-blocks ^
  --no_banner
```

Expected outputs were generated:

- `Factor_A_SNPs.csv`
- `Factor_B_Tissues.csv`
- `Factor_C_Phenotypes.csv`
- `qc_report.json`
- `qc_summary.csv`
- `qc_report.tsv`
- `qc_report.txt`
- `rank_diagnostics.csv`
- `rank_selection.json`

Rank consistency check:

```bash
.venv\Scripts\python.exe tune_rank.py ^
  --manifest examples/mini_fixture_1000/manifest.csv ^
  --bed examples/mini_fixture_1000/ld_blocks_header.bed ^
  --max_rank 5 ^
  --out results/mini_fixture_rank ^
  --no_banner ^
  --seed 0
```

Both `run_prisma.py --rank auto` and `tune_rank.py` selected rank 2 on the mini-fixture.

PLINK/LD CLI failure check:

```bash
.venv\Scripts\python.exe run_prisma.py ^
  --manifest examples/mini_fixture_1000/manifest.csv ^
  --out results/should_fail ^
  --rank 2 ^
  --ld-reference-mode plink ^
  --no_banner
```

Result: failed clearly because `--bfile` was missing.

PLINK/LD end-to-end tiny reference check:

```bash
.venv\Scripts\python.exe run_prisma.py ^
  --manifest examples/mini_fixture_1000/manifest.csv ^
  --out results/mini_fixture_plink_test ^
  --rank 2 ^
  --phenotype-name synthetic_trait ^
  --bfile results/tiny_plink_ref/tiny_ref ^
  --ld-reference-mode plink ^
  --iter 1 ^
  --quiet-blocks ^
  --no_banner
```

Result: passed with `fraction_snps_with_empirical_ld = 1.0`.

## Remaining Limitations

- No SMR, coloc, TWAS, or MAGMA benchmark modules were added in this phase.
- Real eQTL data are not redistributed.
- Identity LD mode is for smoke testing and diagnostics only.
- Full biological replication requires real GWAS, eQTL, LD-block, and PLINK/LD reference inputs.
- The PLINK/LD interface was tested with a deterministic tiny reference. Users still need build-matched real reference panels for production analyses.

## Final Readiness Judgment

READY FOR LD-AWARE REAL-DATA TEST

The public repository now supports dependency self-checks, deterministic smoke testing, schema/QC reports, explicit LD-reference modes, phenotype naming, shared rank selection, and empirical PLINK LD graph construction. The next phase can add benchmark modules without relying on silent identity-LD fallback or hard-coded DR-specific assumptions.
