# Engineering Hardening Final Audit

## Branch

- Branch: `engineering_hardening_pre_benchmarks`
- Audit base commit: `d334a18`

## Files Modified in This Final Pass

- `.github/workflows/tests.yml`
- `README.md`
- `builder.py`
- `run_prisma.py`
- `tests/test_cli_smoke_mini_fixture.py`
- `tests/test_identity_ld_policy.py`
- `tests/test_partition_bed_header.py`
- `tests/test_rank_selection_consistency.py`
- `tune_rank.py`
- `ENGINEERING_HARDENING_FINAL_AUDIT.md`

No manuscript files, figures, supplementary tables, Zenodo packages, restricted GWAS/eQTL data, or scientific benchmark modules were modified in this pass.

## Blocker Fixes Checked

| Issue | Status | Evidence |
|---|---|---|
| pyarrow dependency | pass | `pip install -r requirements.txt` reported `pyarrow>=14.0` already satisfied; `scripts/check_environment.py` imported pyarrow. |
| environment.yml valid | pass | Parsed manually for pip subsection nesting; `bed-reader>=0.2.40` and `mygene>=3.2` are nested under `- pip:`. |
| BED header/no-header parsing | pass | `pytest` passed BED parser tests for headered, no-header, whitespace, invalid structure, and comma no-header cases. No fixed row-skipping parser remains. |
| LD reference CLI exposed | pass | `run_prisma.py` exposes `--bfile`, `--ld-reference-mode`, `--allow-identity-ld`, LD overlap and coverage thresholds. |
| identity LD requires explicit permission | pass | `--ld-reference-mode identity` without `--allow-identity-ld` fails; real-data-style `auto` without `--bfile` fails; examples/tests auto identity remains allowed and recorded. |
| phenotype-name generalized | pass | `--phenotype-name height` is tested; `Factor_C_Phenotypes.csv` uses the selected phenotype name. |
| QC reports generated | pass | Smoke run produced `qc_report.json`, `qc_summary.csv`, `qc_report.tsv`, and `qc_report.txt` before factorization. |
| rank selection unified | pass | `run_prisma.py --rank auto` and `tune_rank.py` both call the shared `select_rank` function; CLI consistency test passes. |
| mini-fixture smoke test | pass | Deterministic synthetic 1000-SNP fixture generated and completed PRISMA smoke run. |
| PLINK empirical LD support | pass | Tiny PLINK smoke run reported `fraction_snps_with_empirical_ld=0.04` and `n_blocks_empirical_laplacian=1`. |
| GitHub Actions CI added | pass | `.github/workflows/tests.yml` runs dependency install, environment check, and `pytest -q` on push and pull requests. |

## Commands Run

| Command | Result |
|---|---|
| `.venv\Scripts\python.exe -m pip install -r requirements.txt` | pass |
| `.venv\Scripts\python.exe scripts\check_environment.py` | pass |
| PowerShell/Python manual parse of `environment.yml` pip subsection | pass |
| `.venv\Scripts\python.exe -m pytest -q` | pass, `18 passed in 27.42s` |
| `.venv\Scripts\python.exe examples\generate_mini_fixture_1000.py` | pass |
| `.venv\Scripts\python.exe run_prisma.py --manifest examples/mini_fixture_1000/manifest.csv --out results/mini_fixture_test --rank auto --max-rank 5 --phenotype-name synthetic_trait --ld-reference-mode identity --allow-identity-ld --iter 5 --quiet-blocks` | pass |
| `.venv\Scripts\python.exe run_prisma.py --manifest examples/mini_fixture_1000/manifest.csv --out results/should_fail_no_identity_permission --rank 2 --phenotype-name synthetic_trait --ld-reference-mode identity --iter 2 --quiet-blocks` | expected failure observed |
| `.venv\Scripts\python.exe run_prisma.py --manifest examples/mini_fixture_1000/manifest.csv --out results/should_fail_plink_no_bfile --rank 2 --phenotype-name synthetic_trait --ld-reference-mode plink --iter 2 --quiet-blocks` | expected failure observed |
| `.venv\Scripts\python.exe run_prisma.py --manifest results/real_style_manifest_for_auto_fail.csv --out results/should_fail_auto_real_no_bfile --rank 2 --phenotype-name synthetic_trait --ld-reference-mode auto --iter 1 --quiet-blocks` | expected failure observed |
| Tiny PLINK smoke run with `--ld-reference-mode plink --bfile results/tiny_plink_ref/tiny_ref --allow-low-coverage` | pass, empirical LD coverage recorded as greater than zero |

## Remaining Limitations

- No SMR, coloc, TWAS, MAGMA, or other scientific benchmark module was added in this phase.
- The mini-fixture is deterministic synthetic data and is not biologically interpretable.
- Real biological interpretation requires real GWAS, eQTL, LD-block, and PLINK/LD reference inputs with consistent genome build and allele conventions.
- Identity LD mode is for synthetic smoke tests or diagnostics only and does not reproduce the manuscript LD-aware model.
- High-quality, trait-relevant multi-tissue eQTL data remain necessary for biological interpretation.

## Final Readiness Judgment

READY TO MERGE TO MAIN

All release-blocking engineering checks passed locally: dependency self-check, full pytest suite, synthetic mini-fixture smoke test, explicit identity-LD policy, expected failure paths, QC report generation, rank-selection consistency, and empirical PLINK LD smoke test.
