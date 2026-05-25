# PRISMA

PRISMA (Polygenic Risk Integration via Summary-statistics Multi-tissue Array-decomposition) decomposes GWAS summary statistics and multi-tissue eQTL evidence into interpretable tissue-anchored polygenic axes.

This public release currently contains the core computational engine only. Raw GWAS/eQTL files, single-cell data, vitreous proteomics/metabolomics data, manuscript figures, intermediate results, and deprecated exploratory scripts are intentionally excluded.

## Core Files

- run_prisma.py: main command-line entry point.
- loader.py: GWAS/eQTL manifest loading, genomic-control scaling, allele alignment, and SMR-style summary statistic construction.
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

## Genome Build Recommendation

For manuscript-style analyses, we recommend using GRCh37/hg19 coordinates. The diabetic retinopathy analysis used GRCh37/hg19-aligned GWAS and eQTL inputs, LDetect European LD blocks in hg19 coordinates, and the 1000 Genomes Project Phase 3 European reference panel on the same coordinate system.

PRISMA can be applied to other genome builds, but all inputs must use a consistent build. If using GRCh38/hg38 data, lift over or replace every coordinate-dependent input accordingly, including the GWAS summary statistics, eQTL files, LD-block BED file, gene-position annotations, and any optional PLINK/bed-reader LD reference panel. Mixing hg19 and hg38 resources can change LD-block assignment, SNP matching, representative selection, and downstream factor loadings.

## GWAS QC Recommendation

Before running PRISMA, we recommend excluding the major histocompatibility complex (MHC) region and other known complex-LD regions from the trait GWAS summary statistics during GWAS quality control. The diabetic retinopathy manuscript analysis used an MHC-excluded GWAS input. Although the LD-block partitioner can exclude several complex-LD blocks when an appropriate LD-block BED file is supplied, the targeted gene blacklist is not intended to replace GWAS-level MHC filtering. Removing MHC before PRISMA reduces the risk that long-range LD structure or locus-level LD traps dominate the factorization.

## Quick Start

The repository includes a tiny fully synthetic example dataset in examples/synthetic_data. To regenerate it deterministically:

    python examples/generate_synthetic_example.py

Run PRISMA from the repository root:

    python run_prisma.py --manifest examples/synthetic_data/manifest.csv --out results/synthetic_demo --rank 3 --iter 5

For your own data, provide a manifest using the same structure:

    python run_prisma.py --manifest examples/data_manifest_template.csv --out results --rank 3 --iter 20

Output files:

- Factor_A_SNPs.csv: SNP/locus factor loadings.
- Factor_B_Tissues.csv: tissue-mode factor loadings.
- Factor_C_Phenotypes.csv: phenotype-mode factor loadings.

Rank exploration:

    python tune_rank.py --manifest examples/synthetic_data/manifest.csv --bed examples/synthetic_data/synthetic_ld_blocks.bed --max_rank 5

## eQTL Preprocessing

The diabetic retinopathy manuscript used SMR-formatted GTEx v8 BESD files for the GTEx tissue eQTL layers. Because GTEx eQTL resources are third-party datasets subject to their original access and use terms, these eQTL files are not redistributed in this repository. Users can prepare PRISMA-formatted eQTL inputs from SMR-formatted GTEx BESD resources or from official GTEx Portal association tables using the preprocessing rules below. In our Whole Blood sensitivity analysis, replacing the SMR-derived eQTL layer with a raw GTEx-derived layer preserved the three tissue-level PRISMA axes after rank alignment, indicating that the main tissue-axis conclusions are not driven by SMR formatting.

The `scripts/` folder includes a utility for converting raw GTEx eQTL association tables to PRISMA-formatted eQTL inputs:

    python scripts/clean_gtex_eqtl_for_prisma.py \
      --input data/raw/Whole_Blood.tsv.gz \
      --output data/Whole_Blood_eQTL_cleaned_gtex_raw.txt \
      --epi-map data/raw/Whole_Blood.lite.epi \
      --summary-output results/Whole_Blood_cleaning_summary.csv

Raw GTEx eQTL tables can contain multiple gene-level associations for the same rsID. PRISMA uses a SNP x tissue input matrix, so this preprocessing step retains one representative association per SNP: the row with the largest absolute eQTL Z-score, where Z = beta / se. For GTEx variant records, beta is interpreted with respect to ALT, so the standard output uses A1=ALT and A2=REF.

## Installation

A minimal Python environment is sufficient for the core engine:

    pip install -r requirements.txt

Optional LD reference-panel support in builder.py uses bed-reader. If no PLINK reference panel is supplied, the code falls back to an identity Laplacian within each block.

If your eQTL target genes are Ensembl IDs, `mygene>=3.2` is required for manuscript-style exact filtering. PRISMA uses `mygene` to map Ensembl IDs to gene symbols before applying the targeted housekeeping-gene and 17q21.31 LD-trap filters documented in `resources/targeted_gene_blacklist.tsv`. This mapping step is skipped for gene-symbol inputs and for the synthetic example data.

## Public-Release Notes

This folder was prepared from an internal research workspace. The internal workspace contains raw data, manuscript-specific result folders, plotting scripts, and archived/deprecated exploratory analyses. Those materials are not included here because they either depend on controlled/local data, contain hard-coded private paths, or are not part of the reusable PRISMA core engine.

The PRISMA source code in this repository is released under the Apache License, Version 2.0. See LICENSE and NOTICE. Third-party datasets used in the accompanying manuscript are not redistributed and remain subject to their original data-access terms.

PRISMA is research software and is not intended for clinical diagnosis or treatment decisions.

## Citation

If you use PRISMA, please cite the accompanying manuscript/preprint once available.
