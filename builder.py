import numpy as np
import pandas as pd

class TensorBuilder:
    def __init__(
        self,
        n_tissues,
        n_phenotypes,
        rho_matrix=None,
        N_eff=None,
        bfile_path=None,
        ld_reference_mode="auto",
        ld_min_overlap=2,
        quiet_blocks=False,
    ):
        self.n_tissues = n_tissues
        self.n_phenotypes = n_phenotypes
        # These column lists are supplied by the caller.
        self.tissue_cols = []
        self.pheno_cols = []
        # Sample-overlap correction parameters.
        self.rho_matrix = rho_matrix  # [Tissues, Phenos]
        self.N_eff = N_eff
        # Optional LD reference panel prefix.
        self.bfile_path = bfile_path
        self.ld_reference_mode = ld_reference_mode
        self.ld_min_overlap = int(ld_min_overlap)
        self.quiet_blocks = quiet_blocks
        self._snp_map = None  # Cached SNP ID -> index map.
        self.laplacian_stats = {
            "ld_reference_mode": ld_reference_mode,
            "bfile_path": bfile_path,
            "n_blocks_identity_laplacian": 0,
            "n_blocks_empirical_laplacian": 0,
            "n_blocks_dropped_insufficient_snps": 0,
            "n_laplacian_calls": 0,
            "n_snps_seen_for_laplacian": 0,
            "n_snps_with_reference_overlap": 0,
            "last_block_modes": [],
        }

        # Read the .bim file during initialization when a reference panel is supplied.
        if self.bfile_path is not None:
            bim_file = f"{self.bfile_path}.bim"
            try:
                bim_df = pd.read_csv(bim_file, sep=r"\s+", header=None, usecols=[1], engine="python")
                self._snp_map = {snp: idx for idx, snp in enumerate(bim_df[1].values)}
                print(f"[INFO] Loaded PLINK BIM SNP map: {len(self._snp_map)} variants from {bim_file}")
            except Exception as e:
                raise ValueError(f"Could not read PLINK .bim file for LD reference: {bim_file}") from e

    def get_reference_overlap_count(self, snp_list):
        if self._snp_map is None:
            return 0
        return sum(1 for snp in snp_list if snp in self._snp_map)

    def build_tensor(self, block_df):
        """
        Build tensor X [SNPs, Tissues, Phenotypes] from PRISMA integration scores
        already computed by the loader.
        """
        n_snps = len(block_df)
        X = np.zeros((n_snps, self.n_tissues, self.n_phenotypes))

        # Extract PRISMA integration scores.
        for t_idx, t_col in enumerate(self.tissue_cols):
            X[:, t_idx, 0] = block_df[t_col].values

        # Optional sample-overlap correction.
        if self.rho_matrix is not None and self.N_eff is not None:
            correction = self.rho_matrix * np.sqrt(self.N_eff)
            X -= correction[np.newaxis, :, :]

        # Variance-stabilizing transformation.
        X = np.sign(X) * np.sqrt(np.abs(X))
        np.clip(X, -20, 20, out=X)

        return X

    def _get_snp_indices(self, snp_list):
        """
        Return SNP indices in the reference panel and a mask of matched SNPs.
        """
        found_indices = []
        found_mask = np.zeros(len(snp_list), dtype=bool)

        for i, snp in enumerate(snp_list):
            if snp in self._snp_map:
                found_indices.append(self._snp_map[snp])
                found_mask[i] = True

        return found_indices, found_mask

    def build_laplacian(self, block_df):
        """
        Build graph Laplacian L = D - W using LD correlations from an optional
        reference panel. If no reference panel is available, use identity.
        """
        n_snps = len(block_df)
        self.laplacian_stats["n_laplacian_calls"] += 1
        self.laplacian_stats["n_snps_seen_for_laplacian"] += int(n_snps)

        # Fallback: no reference panel, so SNPs are treated as independent.
        if self.bfile_path is None or self._snp_map is None:
            self.laplacian_stats["n_blocks_identity_laplacian"] += 1
            self.laplacian_stats["last_block_modes"].append("identity_no_reference")
            return np.eye(n_snps)

        # Map SNPs to reference-panel indices.
        snp_list = block_df['SNP'].values
        found_indices, found_mask = self._get_snp_indices(snp_list)
        self.laplacian_stats["n_snps_with_reference_overlap"] += int(len(found_indices))

        # Initialize as identity; unmatched SNPs remain independent.
        L = np.eye(n_snps)

        if len(found_indices) < self.ld_min_overlap:
            self.laplacian_stats["n_blocks_identity_laplacian"] += 1
            self.laplacian_stats["n_blocks_dropped_insufficient_snps"] += 1
            self.laplacian_stats["last_block_modes"].append("identity_insufficient_overlap")
            if not self.quiet_blocks:
                print(
                    "[WARNING] LD block used identity Laplacian "
                    f"because reference overlap was {len(found_indices)} "
                    f"(< {self.ld_min_overlap})."
                )
            return L

        # Read genotype matrix.
        try:
            from bed_reader import open_bed
            bed_file = f"{self.bfile_path}.bed" if not str(self.bfile_path).endswith(".bed") else self.bfile_path
            with open_bed(bed_file) as bed:
                try:
                    G = bed.read(index=np.s_[:, found_indices])  # [N_individuals, M_found_snps]
                except Exception:
                    G = bed.read(index=found_indices)
        except Exception as e:
            raise ValueError(f"Could not read PLINK .bed file for LD reference: {bed_file}") from e

        # Mean-impute missing genotypes.
        col_means = np.nanmean(G, axis=0)
        for j in range(G.shape[1]):
            mask = np.isnan(G[:, j])
            if mask.any():
                G[mask, j] = col_means[j]

        # Correlation matrix.
        with np.errstate(invalid='ignore'):
            R = np.corrcoef(G, rowvar=False)

        R = np.nan_to_num(R, nan=0.0)

        # Adjacency matrix W = |R|.
        W_sub = np.abs(R)
        np.fill_diagonal(W_sub, 0)

        # Laplacian matrix.
        D_sub = np.diag(W_sub.sum(axis=1))
        L_sub = D_sub - W_sub

        # Fill matched SNPs back into the full matrix.
        found_idx_list = np.where(found_mask)[0]
        for i, idx_i in enumerate(found_idx_list):
            for j, idx_j in enumerate(found_idx_list):
                L[idx_i, idx_j] = L_sub[i, j]

        self.laplacian_stats["n_blocks_empirical_laplacian"] += 1
        self.laplacian_stats["last_block_modes"].append("empirical")
        if not self.quiet_blocks:
            print(f"[INFO] LD block used empirical LD graph for {len(found_indices)}/{n_snps} SNPs.")
        return L

    def summarize_laplacian_usage(self):
        stats = dict(self.laplacian_stats)
        seen = max(int(stats.get("n_snps_seen_for_laplacian", 0)), 1)
        stats["fraction_snps_with_empirical_ld"] = float(stats.get("n_snps_with_reference_overlap", 0) / seen)
        return stats
