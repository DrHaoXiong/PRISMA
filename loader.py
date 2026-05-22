import polars as pl
import pandas as pd
import numpy as np
import os

class DataLoader:
    """
    Minimal DataFrame loader kept for backward-compatible utility workflows.
    """
    def load_and_align(self, df_gwas, df_eqtl):
        """
        Basic allele alignment.
        """
        # Merge data.
        df_merged = df_gwas.merge(df_eqtl, on='SNP', how='inner', suffixes=('', '_eqtl'))

        # Basic alignment: matched or flipped alleles.
        match = (df_merged['A1'] == df_merged['A1_eqtl']) & (df_merged['A2'] == df_merged['A2_eqtl'])
        flip = (df_merged['A1'] == df_merged['A2_eqtl']) & (df_merged['A2'] == df_merged['A1_eqtl'])

        df_aligned = df_merged[match | flip].copy()

        # Flip eQTL BETA when alleles are reversed.
        flip_mask = df_aligned['A1'] == df_aligned['A2_eqtl']
        if 'BETA_eqtl' in df_aligned.columns:
            df_aligned.loc[flip_mask, 'BETA_eqtl'] *= -1

        return df_aligned

class TensorDataLoader:
    """
    PRISMA tensor data loader using Polars.

    Responsibilities:
    1. Lazily load GWAS and multi-tissue eQTL summary statistics.
    2. Apply optional genomic-control scaling.
    3. Align alleles across GWAS and eQTL files.
    4. Assemble standardized tensor-ready input.
    """

    def __init__(self, manifest_path, apply_genomic_control=True):
        """
        Initialize loader.

        Parameters:
        - manifest_path: path to the input manifest CSV.
        - apply_genomic_control: whether to apply genomic-control scaling.
        """
        self.manifest_path = manifest_path
        self.apply_genomic_control = apply_genomic_control

        # Read manifest.
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

        self.manifest = pd.read_csv(manifest_path)
        self.stats = {}
        print(f"[INFO] Reading data manifest: {manifest_path}")
        print(f"       Found {len(self.manifest)} data files.")

    def _compute_lambda_gc(self, z_scores):
        """
        Compute the genomic-control inflation factor:
        lambda_GC = median(Z^2) / 0.4549.
        """
        z_squared = z_scores ** 2
        lambda_gc = z_squared.median() / 0.4549
        return lambda_gc

    def _apply_gc_scaling(self, lf, z_col='Z'):
        """
        Apply genomic-control scaling to a LazyFrame.
        If lambda_GC > 1.0, use Z_scaled = Z / sqrt(lambda_GC).
        """
        # Collect once to compute the median-based inflation factor.
        z_values = lf.select(z_col).collect()[z_col]
        lambda_gc = float(self._compute_lambda_gc(z_values))

        if lambda_gc > 1.0:
            print(f"      lambda_GC = {lambda_gc:.3f} (scaling applied)")
            lf = lf.with_columns(
                (pl.col(z_col) / np.sqrt(lambda_gc)).alias(z_col)
            )
        else:
            print(f"      lambda_GC = {lambda_gc:.3f} (no scaling needed)")

        return lf

    def _align_alleles(self, lf_target, lf_ref, tissue_name):
        """
        Align target eQTL alleles to the reference GWAS backbone.

        Logic:
        - Match: Z_MR = sign(Z_GWAS * Z_eQTL) * abs_smr
        - Flip:  Z_MR = -sign(Z_GWAS * Z_eQTL) * abs_smr
        - Mismatch: set the tissue-specific statistic to zero.

        abs_smr = sqrt((Z_GWAS^2 * Z_eQTL^2) / (Z_GWAS^2 + Z_eQTL^2 + epsilon))
        """
        # Join on SNP while preserving all GWAS backbone variants.
        lf_joined = lf_ref.join(lf_target, on='SNP', how='left', suffix='_target')

        # Allele-alignment flags.
        lf_aligned = lf_joined.with_columns([
            ((pl.col('A1') == pl.col('A1_target')) &
             (pl.col('A2') == pl.col('A2_target'))).alias('match'),
            ((pl.col('A1') == pl.col('A2_target')) &
             (pl.col('A2') == pl.col('A1_target'))).alias('flip')
        ])

        # Compute SMR-style statistics.
        z_gwas = pl.col('GWAS_Z')
        z_eqtl = pl.col('Z_eqtl')

        # Unsigned SMR-style strength.
        abs_smr = (
            ((z_gwas ** 2) * (z_eqtl ** 2)) /
            ((z_gwas ** 2) + (z_eqtl ** 2) + 1e-10)
        ).sqrt()

        # Sign of the matched GWAS-eQTL product.
        sign_product = (z_gwas * z_eqtl).sign()

        # Apply alignment: match keeps sign, flip reverses sign, mismatch is zero.
        lf_aligned = lf_aligned.with_columns([
            pl.when(pl.col('match'))
              .then(sign_product * abs_smr)
              .when(pl.col('flip'))
              .then(-sign_product * abs_smr)
              .otherwise(pl.lit(0.0))
              .alias(f'{tissue_name}_Z'),
            pl.col('TARGET_GENE')
              .fill_null(pl.lit('no_eqtl'))
              .alias(f'{tissue_name}_GENE')
        ])

        # Keep only required columns.
        schema_names = lf_aligned.collect_schema().names()
        cols_to_keep = ['SNP', 'CHR', 'BP', 'A1', 'A2', 'GWAS_Z']
        cols_to_keep += [c for c in schema_names if (c.endswith('_Z') or c.endswith('_GENE')) and c != 'GWAS_Z']
        lf_aligned = lf_aligned.select([c for c in cols_to_keep if c in schema_names])

        return lf_aligned

    def load_and_align(self):
        """
        Main loading workflow:
        1. Load GWAS as the backbone.
        2. Iterate over eQTL files with GC scaling and allele alignment.
        3. Join all tissue-specific statistics.
        """
        # 1. Extract GWAS file.
        gwas_files = self.manifest[self.manifest['type'] == 'gwas']
        if len(gwas_files) == 0:
            raise ValueError("Manifest does not contain a GWAS file.")

        gwas_path = gwas_files.iloc[0]['path']
        gwas_name = gwas_files.iloc[0]['name']
        print(f"\n[INFO] Building GWAS backbone: {gwas_name}")

        # Lazily load GWAS. CHR may include X/Y/MT, so read it as string first.
        lf_gwas = pl.scan_csv(
            gwas_path,
            separator='\t',
            schema_overrides={'CHR': pl.Utf8}
        )

        # Standardize column names.
        lf_gwas = lf_gwas.rename({
            'effect_allele': 'A1',
            'other_allele': 'A2',
            'beta': 'BETA',
            'se': 'SE',
            'pval': 'P'
        })

        # Keep autosomes only.
        lf_gwas = lf_gwas.filter(
            pl.col('CHR').cast(pl.Int32, strict=False).is_not_null()
        )

        # Convert CHR to integer.
        lf_gwas = lf_gwas.with_columns(
            pl.col('CHR').cast(pl.Int32).alias('CHR')
        )

        # Compute GWAS Z-score.
        lf_gwas = lf_gwas.with_columns(
            (pl.col('BETA') / (pl.col('SE') + 1e-10)).alias('GWAS_Z')
        ).select(['SNP', 'CHR', 'BP', 'A1', 'A2', 'GWAS_Z'])

        # 2. Extract eQTL files.
        eqtl_files = self.manifest[self.manifest['type'] == 'eqtl']
        if len(eqtl_files) == 0:
            raise ValueError("Manifest does not contain any eQTL files.")

        print(f"\n[INFO] Aligning {len(eqtl_files)} eQTL data sources...")

        # 3. Process each eQTL file.
        lf_backbone = lf_gwas

        for idx, row in eqtl_files.iterrows():
            tissue_name = row['name']
            tissue_path = row['path']
            print(f"\n   Processing tissue: {tissue_name}")

            # Lazily load eQTL.
            lf_eqtl = pl.scan_csv(tissue_path, separator='\t')

            # Compute raw eQTL Z-score.
            lf_eqtl = lf_eqtl.with_columns(
                (pl.col('BETA') / (pl.col('SE') + 1e-10)).alias('Z')
            )

            # Apply genomic-control scaling to the eQTL Z-score.
            if self.apply_genomic_control:
                z_values = lf_eqtl.select('Z').collect()['Z']
                lambda_gc = float(self._compute_lambda_gc(z_values))
                if lambda_gc > 1.0:
                    print(f"      lambda_GC = {lambda_gc:.3f} (scaling applied)")
                    lf_eqtl = lf_eqtl.with_columns(
                        (pl.col('Z') / np.sqrt(lambda_gc)).alias('Z_eqtl')
                    )
                else:
                    print(f"      lambda_GC = {lambda_gc:.3f} (no scaling needed)")
                    lf_eqtl = lf_eqtl.with_columns(
                        pl.col('Z').alias('Z_eqtl')
                    )
            else:
                lf_eqtl = lf_eqtl.with_columns(
                    pl.col('Z').alias('Z_eqtl')
                )

            # Keep required columns.
            lf_eqtl = lf_eqtl.select(['SNP', 'A1', 'A2', 'Z_eqtl', 'TARGET_GENE'])

            # Prevent cartesian expansion: keep the strongest eQTL gene per SNP.
            lf_eqtl = (
                lf_eqtl
                .with_columns(pl.col('Z_eqtl').abs().alias('_abs_z'))
                .sort('_abs_z', descending=True)
                .unique(subset=['SNP'], keep='first')
                .drop('_abs_z')
            )

            # Align alleles.
            lf_backbone = self._align_alleles(lf_eqtl, lf_backbone, tissue_name)

        # 4. Materialize the final joined table.
        print("\n[INFO] Executing lazy computation and merging data...")
        df_final = lf_backbone.collect()

        print(f"   Raw dimensions: {df_final.shape}")

        # ===== SMR-style gene representative pruning =====
        # Keep one SNP per target gene using the strongest cross-tissue eQTL signal.
        # This follows the SMR convention of using a top cis-eQTL instrument.
        gene_cols = [c for c in df_final.columns if c.endswith('_GENE')]
        tissue_z_cols = [c for c in df_final.columns if c.endswith('_Z') and c != 'GWAS_Z']

        if gene_cols and tissue_z_cols:
            # Step 0: remove GWAS-only rows with no tissue eQTL support.
            has_eqtl = pl.lit(False)
            for gc in gene_cols:
                has_eqtl = has_eqtl | (pl.col(gc) != 'no_eqtl')

            n_total = len(df_final)
            df_final = df_final.filter(has_eqtl)
            self.stats['N_Raw_Backbone_Variants'] = int(n_total)
            self.stats['N_Candidate_Variants_PreLD'] = int(len(df_final))
            print(f"   Polars prefilter: {n_total} -> {len(df_final)} "
                  f"(removed {n_total - len(df_final)} GWAS-only SNPs)")

            # Step 1: infer a per-row consensus gene.
            gene_pd = df_final.select(gene_cols).to_pandas()
            gene_pd = gene_pd.replace('no_eqtl', pd.NA)
            consensus_gene = gene_pd.mode(axis=1)[0]

            # Step 1.5: defensively remove rows whose consensus gene remains missing.
            valid_mask = consensus_gene.notna()
            if not valid_mask.all():
                df_final = df_final.filter(pl.Series(valid_mask.values))
                consensus_gene = consensus_gene[valid_mask].reset_index(drop=True)
                print(f"   Removed residual invalid rows: {len(df_final)} rows remain")

            df_final = df_final.with_columns(
                pl.Series('_consensus_gene', consensus_gene.values)
            )

            # Step 2: cross-tissue composite eQTL strength = max(|Z_tissue|).
            z_pd = df_final.select(tissue_z_cols).to_pandas().abs()
            composite_z = z_pd.max(axis=1)
            df_final = df_final.with_columns(
                pl.Series('_composite_z', composite_z.values)
            )

            n_before = len(df_final)

            # Step 3: keep the strongest SNP for each consensus gene.
            df_final = (
                df_final
                .sort('_composite_z', descending=True)
                .group_by('_consensus_gene')
                .first()
                .sort(['CHR', 'BP'])
            )

            n_after = len(df_final)
            self.stats['N_Gene_Representatives_PreBlacklist'] = int(n_after)
            print(f"   SMR-style gene representative pruning: {n_before} SNPs -> {n_after} gene representatives")
            print(f"   Compression rate: {(1 - n_after/n_before)*100:.1f}%")

            # =========================================================
            # Targeted cleanup: housekeeping genes and known LD-trap proxies.
            # =========================================================
            # 1. Housekeeping genes.
            housekeeping_prefixes = ('RPS', 'RPL', 'EEF', 'EIF')

            # 2. 17q21.31 inversion-region LD proxies.
            ld_trap_genes = {'KANSL1-AS1', 'ARHGAP27', 'LRRC37A2', 'ARL17A', 'MAPT', 'CRHR1', 'SPPL2C', 'STH'}

            gene_series = df_final['_consensus_gene'].to_list()
            # Remove optional Ensembl version suffix (ENSG00000123.4 -> ENSG00000123).
            clean_ensgs = [str(g).split('.')[0] for g in gene_series]
            
            # Map Ensembl IDs to gene symbols only when needed; synthetic symbols skip network calls.
            if any(str(g).startswith('ENSG') for g in clean_ensgs):
                print("   Detected Ensembl IDs. Querying mygene for symbol-based blacklist filtering...")
                try:
                    import mygene
                    mg = mygene.MyGeneInfo()
                    # verbose=False keeps logs compact.
                    results = mg.querymany(clean_ensgs, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)
                    symbol_dict = {res['query']: str(res.get('symbol', '')).upper() for res in results if 'symbol' in res}
                except ImportError:
                    print("   [WARNING] mygene is not installed. Symbol-based blacklist filtering may be incomplete.")
                    symbol_dict = {}
                except Exception as e:
                    print(f"   [WARNING] mygene query failed: {e}")
                    symbol_dict = {}
            else:
                symbol_dict = {}

            # Apply blacklist filter.
            keep_mask = []
            for ensg in clean_ensgs:
                g_sym = symbol_dict.get(ensg, ensg)
                
                is_hk = g_sym.startswith(housekeeping_prefixes)
                is_trap = g_sym in ld_trap_genes
                
                keep_mask.append(not (is_hk or is_trap))

            n_before_filter = len(df_final)
            df_final = df_final.filter(pl.Series(keep_mask))
            n_removed = n_before_filter - len(df_final)

            if n_removed > 0:
                print(f"   Targeted cleanup removed {n_removed} housekeeping or 17q21 LD-trap genes.")
            else:
                print("   Targeted cleanup detected no housekeeping or 17q21 LD-trap genes.")

            # Drop temporary columns.
            df_final = df_final.drop(['_consensus_gene', '_composite_z'])

        print(f"[INFO] Loading complete. Final dimensions: {df_final.shape}")
        print(f"       Columns: {df_final.columns}")
        self.stats['N_Final_Variants'] = int(len(df_final))

        return df_final.to_pandas()
