# PRISMA v0.2.0 Peer-Review Core Release Notes

This release corresponds to the public PRISMA core code available during peer
review of the diabetic retinopathy manuscript. It contains the core LD-aware
PRISMA engine, QC utilities, rank diagnostics, synthetic tests, reproducibility
helpers, and public documentation.

## Included Scope

- Core PRISMA command-line workflow for the validated single-phenotype public
  pipeline.
- Empirical LD reference policy for real-data analyses through `--bfile`.
- Explicit identity-LD boundary for synthetic smoke tests and diagnostics.
- QC reports for schema, allele harmonization, tensor coverage, LD block
  assignment, and postfit Laplacian usage.
- Shared rank-selection logic and seed-stability utility.
- In-memory empirical-LD Laplacian caching and reused PLINK bed-reader handles
  to avoid recomputing identical block Laplacians across ALS epochs.

## Data Scope

Manuscript-associated derived data are archived separately on Zenodo:

- Current companion package: https://zenodo.org/records/20535036
- DOI: `10.5281/zenodo.20535036`

Restricted raw GWAS/eQTL inputs, individual-level data, raw single-cell
matrices, and controlled-access resources are not redistributed in this source
repository.

## LD Provenance Requirement

For real-data LD-aware PRISMA analyses, archive all of the following:

- exact `run_prisma.py` command;
- PLINK reference prefix and SHA256 hashes for `.bed`, `.bim`, and `.fam`;
- `qc_report.json`;
- `qc_summary.csv`;
- postfit empirical-LD usage fields, including empirical versus identity
  Laplacian block counts and fraction of SNPs with empirical LD.

Runtime commands that omit `--bfile` should be interpreted as identity-LD or
provenance-limited runs, not as evidence of empirical PLINK-LD analysis.
