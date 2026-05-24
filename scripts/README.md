# eQTL preprocessing utilities

This folder contains optional utilities for preparing eQTL summary statistics for PRISMA.

## Raw GTEx eQTL tables

`clean_gtex_eqtl_for_prisma.py` converts a raw GTEx eQTL association table, such as `Whole_Blood.tsv.gz`, into the PRISMA eQTL input schema:

```text
SNP, A1, A2, BETA, SE, CHR, BP, TARGET_GENE, P
```

Raw GTEx tables can contain multiple gene-level associations for the same rsID. PRISMA uses a SNP x tissue matrix, so the script keeps one representative association per SNP: the row with the largest absolute eQTL Z-score, where `Z = beta / se`.

Example:

```bash
python scripts/clean_gtex_eqtl_for_prisma.py \
  --input data/raw/Whole_Blood.tsv.gz \
  --output data/Whole_Blood_eQTL_cleaned_gtex_raw.txt \
  --epi-map data/raw/Whole_Blood.lite.epi \
  --summary-output results/Whole_Blood_cleaning_summary.csv
```

The `--epi-map` argument is optional. It is used only to map Ensembl gene IDs to gene symbols when an SMR `.epi` file is available.

## Exact reproduction versus robustness reproduction

For exact reproduction of a manuscript run, use the cleaned eQTL inputs distributed with the accompanying data package. Re-cleaning from raw GTEx may not produce bitwise-identical SNP-level representative eQTL inputs because raw GTEx tables and SMR-formatted GTEx BESD files differ in formatting, coverage, and representative SNP selection.

For robustness checks, users can regenerate one or more eQTL layers from raw GTEx and rerun PRISMA with an updated manifest. In the diabetic retinopathy application, replacing the SMR-derived Whole Blood eQTL layer with a raw GTEx-derived Whole Blood layer preserved the three tissue-level PRISMA axes after rank alignment, supporting robustness of the main tissue-axis conclusions.
