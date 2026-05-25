# PRISMA Resources

This folder contains small, non-sensitive metadata files used to document manuscript-style preprocessing choices.

- `targeted_gene_blacklist.tsv` records the symbol-level housekeeping and 17q21.31 LD-trap filters used by the loader.

When `TARGET_GENE` values are Ensembl IDs, PRISMA uses `mygene` to map Ensembl IDs to gene symbols before applying these filters. This repository does not bundle a full Ensembl-to-symbol mapping table because that would duplicate a third-party annotation snapshot and require separate version maintenance. The blacklist file is a compact record of PRISMA's filtering rules, not a redistributed eQTL/GWAS resource.
