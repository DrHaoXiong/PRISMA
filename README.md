# PRISMA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20340998.svg)](https://doi.org/10.5281/zenodo.20340998)

PRISMA (Polygenic Risk Integration via Summary-statistics Multi-tissue Array-decomposition) decomposes GWAS summary statistics and multi-tissue eQTL evidence into interpretable tissue-anchored polygenic axes.

This public release currently contains the core computational engine only. Raw GWAS/eQTL files, single-cell data, vitreous proteomics/metabolomics data, manuscript figures, intermediate results, and deprecated exploratory scripts are intentionally excluded.

## Reproducibility Data Package

The derived data and reproducibility materials accompanying the PRISMA diabetic retinopathy manuscript are available on Zenodo:

- [PRISMA diabetic retinopathy derived data and reproducibility package](https://doi.org/10.5281/zenodo.20340998)
- DOI: `10.5281/zenodo.20340998`

The Zenodo package contains author-generated derived data, supplementary tables, supplementary data, supplementary notes, and checksums. It does not redistribute restricted raw eQTL files, raw single-cell matrices, individual-level genotype/phenotype data, or controlled-access data.

## Core Files

- run_prisma.py: main command-line entry point.
- loader.py: GWAS/eQTL manifest loading, genomic-control scaling, allele alignment, and PRISMA GWAS-eQTL integration score construction.
- partition.py: LD-block partitioning and block iterator.
- builder.py: tensor and graph Laplacian construction.
- solver.py: graph-regularized block-wise ALS solver.
- tune_rank.py: rank-sensitivity utilities including variance explained and CORCONDIA-style diagnostics.

## Expected Input

PRISMA expects a CSV manifest with columns type, name, and path. See examples/data_manifest_template.csv.

Required row types:

- gwas: tab-separated GWAS summary statistics.
- eqtl: one row per tissue-specific eQTL summary-statistics file.
- bed: LD-block definition file.

Expected GWAS columns:

- SNP, CHR, BP, effect_allele, other_allele, beta, se, pval

Expected eQTL columns:

- SNP, A1, A2, BETA, SE, CHR, BP, TARGET_GENE, P

The core loader uses SNP, A1, A2, BETA, SE, and TARGET_GENE directly. CHR, BP, and P are retained in the standard preprocessing output for traceability and compatibility with manuscript data checks.

Expected LD-block BED columns:

- chr, start, stop

Headered and no-header BED-like files are supported. Delimiters may be tab,
comma, or whitespace. Chromosome labels may use either `chr1` or `1`; PRISMA
normalizes these internally.

## Genome Build Recommendation

For manuscript-style analyses, we recommend using GRCh37/hg19 coordinates. The diabetic retinopathy analysis used GRCh37/hg19-aligned GWAS and eQTL inputs, LDetect European LD blocks in hg19 coordinates, and the 1000 Genomes Project Phase 3 European reference panel on the same coordinate system.

PRISMA can be applied to other genome builds, but all inputs must use a consistent build. If using GRCh38/hg38 data, lift over or replace every coordinate-dependent input accordingly, including the GWAS summary statistics, eQTL files, LD-block BED file, gene-position annotations, and any optional PLINK/bed-reader LD reference panel. Mixing hg19 and hg38 resources can change LD-block assignment, SNP matching, representative selection, and downstream factor loadings.

## GWAS QC Recommendation

Before running PRISMA, we recommend excluding the major histocompatibility complex (MHC) region and other known complex-LD regions from the trait GWAS summary statistics during GWAS quality control. The diabetic retinopathy manuscript analysis used an MHC-excluded GWAS input. Although the LD-block partitioner can exclude several complex-LD blocks when an appropriate LD-block BED file is supplied, the targeted gene blacklist is not intended to replace GWAS-level MHC filtering. Removing MHC before PRISMA reduces the risk that long-range LD structure or locus-level LD traps dominate the factorization.

## Installation

Create an environment and run the dependency self-check:

    pip install -r requirements.txt
    python scripts/check_environment.py

Conda users can alternatively start from:

    conda env create -f environment.yml
    conda activate prisma
    python scripts/check_environment.py

## Quick Smoke Test

The recommended public smoke test uses a deterministic 1000-SNP mini-fixture.
The fixture is synthetic, is not biologically interpretable, and exists only to
test schema handling, allele harmonization, BED parsing, rank selection, QC
reporting, phenotype naming, and output generation.

    python examples/generate_mini_fixture_1000.py

Run PRISMA from the repository root:

    python run_prisma.py \
      --manifest examples/mini_fixture_1000/manifest.csv \
      --out results/mini_fixture_test \
      --rank auto \
      --max-rank 5 \
      --phenotype-name synthetic_trait \
      --ld-reference-mode identity \
      --allow-identity-ld \
      --iter 5

For your own data, provide a manifest using the same structure:

    python run_prisma.py \
      --manifest examples/data_manifest_template.csv \
      --out results/my_trait \
      --rank auto \
      --max-rank 5 \
      --phenotype-name my_trait \
      --bfile /path/to/1000G_EUR_reference \
      --ld-reference-mode plink \
      --iter 20

Output files:

- Factor_A_SNPs.csv: SNP/locus factor loadings.
- Factor_B_Tissues.csv: tissue-mode factor loadings.
- Factor_C_Phenotypes.csv: phenotype-mode factor loadings.
- qc_report.json, qc_summary.csv, qc_report.txt: pre-run QC report.
- rank_diagnostics.csv and rank_selection.json when `--rank auto` is used.

Rank exploration:

    python tune_rank.py \
      --manifest examples/mini_fixture_1000/manifest.csv \
      --bed examples/mini_fixture_1000/ld_blocks_header.bed \
      --max_rank 5 \
      --out results/mini_fixture_rank

`run_prisma.py --rank auto` and `tune_rank.py` call the same rank-selection
logic. The shared rule evaluates ranks 1..max_rank, computes fit and a
CORCONDIA-style diagnostic when possible, selects the lowest non-trivial rank
passing the threshold, and falls back to a variance-explained elbow when no
rank passes.

## LD Reference Mode

For manuscript-style analyses, LD-aware PRISMA requires compatible PLINK binary
reference files (`.bed`, `.bim`, `.fam`) supplied through `--bfile`.

Modes:

- `--ld-reference-mode plink`: require `--bfile` and construct empirical LD
  graphs where SNP overlap is sufficient.
- `--ld-reference-mode auto`: use `--bfile` when supplied. Without `--bfile`,
  identity Laplacians are allowed only for bundled `examples/` or `tests/`
  manifests, or when `--allow-identity-ld` is explicitly supplied.
- `--ld-reference-mode identity --allow-identity-ld`: force identity Laplacians.
  This is intended for synthetic smoke tests and diagnostics only.

Real-data analyses should provide `--bfile` for LD-aware PRISMA. Identity
Laplacian mode is diagnostic and does not reproduce the LD-aware manuscript
model.

## QC Reports

Every run writes QC files before factorization:

- `qc_report.json`: full machine-readable QC report.
- `qc_summary.csv` / `qc_report.tsv`: flattened metric table.
- `qc_report.txt`: concise human-readable summary.

Key fields include:

- `allele_match_rate`: fraction of overlapping GWAS/eQTL SNPs with matched or
  flipped alleles by tissue.
- `fraction_snps_assigned_to_block`: fraction of tensor SNPs assigned to an LD
  block.
- `fraction_snps_with_empirical_ld`: fraction of tensor SNPs present in the
  PLINK reference when `--bfile` is supplied.
- `nonzero_rate_per_tissue`: fraction of retained SNP rows with nonzero
  PRISMA integration score for each tissue.
- `n_blocks_dropped_empty`, `n_blocks_dropped_insufficient_snps`,
  `n_blocks_identity_laplacian`, and `n_blocks_empirical_laplacian`.

Low allele-match, tissue-nonzero, or LD-coverage values should be resolved
before biological interpretation. Override flags are available for diagnostic
workflows, but should not be used to hide input incompatibility.

## eQTL Preprocessing

The diabetic retinopathy manuscript used SMR-formatted GTEx v8 BESD files for the GTEx tissue eQTL layers. Because GTEx eQTL resources are third-party datasets subject to their original access and use terms, these eQTL files are not redistributed in this repository. Users can prepare PRISMA-formatted eQTL inputs from SMR-formatted GTEx BESD resources or from official GTEx Portal association tables using the preprocessing rules below. In our Whole Blood sensitivity analysis, replacing the SMR-derived eQTL layer with a raw GTEx-derived layer preserved the three tissue-level PRISMA axes after rank alignment, indicating that the main tissue-axis conclusions are not driven by SMR formatting.

For primary analyses, we recommend selecting eQTL tissues based on prior biological relevance to the trait rather than indiscriminately including all available GTEx tissues. PRISMA can technically accept many tissue panels, but large panels with many related tissues can introduce strong tissue-mode collinearity, complicate rank selection, and make latent axes harder to interpret. A focused, trait-relevant, non-redundant tissue panel is usually preferable for formal biological interpretation.

The `scripts/` folder includes a utility for converting raw GTEx eQTL association tables to PRISMA-formatted eQTL inputs:

    python scripts/clean_gtex_eqtl_for_prisma.py \
      --input data/raw/Whole_Blood.tsv.gz \
      --output data/Whole_Blood_eQTL_cleaned_gtex_raw.txt \
      --epi-map data/raw/Whole_Blood.lite.epi \
      --summary-output results/Whole_Blood_cleaning_summary.csv

Raw GTEx eQTL tables can contain multiple gene-level associations for the same
rsID. PRISMA uses a SNP × tissue input matrix, so this preprocessing step
retains one representative association per SNP: the row with the largest
absolute eQTL Z-score, where Z = beta / se. For GTEx variant records, beta is
interpreted with respect to ALT, so the standard output uses A1=ALT and A2=REF.

If your eQTL target genes are Ensembl IDs, `mygene>=3.2` is required for manuscript-style exact filtering. PRISMA uses `mygene` to map Ensembl IDs to gene symbols before applying the targeted housekeeping-gene and 17q21.31 LD-trap filters documented in `resources/targeted_gene_blacklist.tsv`. This mapping step is skipped for gene-symbol inputs and for the synthetic example data.

The public repository does not redistribute GTEx or other restricted eQTL
summary statistics. The mini-fixture is for software validation only and is not
biologically interpretable.

## Known Limitations

- LD-aware real-data analysis requires compatible PLINK/LD reference data.
- Identity Laplacian mode is for smoke testing or diagnostics and does not
  reproduce the manuscript LD-aware model.
- High-quality trait-relevant multi-tissue eQTL data remain necessary for
  biological interpretation.
- This repository does not redistribute private or restricted GWAS/eQTL inputs.

## Public-Release Notes

This folder was prepared from an internal research workspace. The internal workspace contains raw data, manuscript-specific result folders, plotting scripts, and archived/deprecated exploratory analyses. Those materials are not included here because they either depend on controlled/local data, contain hard-coded private paths, or are not part of the reusable PRISMA core engine.

The PRISMA source code in this repository is released under the Apache License, Version 2.0. See LICENSE and NOTICE. Third-party datasets used in the accompanying manuscript are not redistributed and remain subject to their original data-access terms.

PRISMA is research software and is not intended for clinical diagnosis or treatment decisions.

## Citation

If you use PRISMA, please cite the accompanying bioRxiv preprint:

Xiong H, Xu W, Ji A, Zhong L, Liu S, Xie Z, Yan J, Wu Z. PRISMA: A tensor-based framework for deconstructing the genetic architecture of complex diseases, with application to diabetic retinopathy. bioRxiv. 2026. doi: [10.64898/2026.05.25.727382](https://doi.org/10.64898/2026.05.25.727382).

Preprint URL: https://www.biorxiv.org/content/10.64898/2026.05.25.727382v1
