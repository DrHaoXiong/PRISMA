import numpy as np
import pandas as pd

class TensorBuilder:
    def __init__(self, n_tissues, n_phenotypes, rho_matrix=None, N_eff=None, bfile_path=None):
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
        self._snp_map = None  # Cached SNP ID -> index map.

        # Read the .bim file during initialization when a reference panel is supplied.
        if self.bfile_path is not None:
            bim_file = f"{self.bfile_path}.bim"
            try:
                bim_df = pd.read_csv(bim_file, sep='\t', header=None, usecols=[1])
                self._snp_map = {snp: idx for idx, snp in enumerate(bim_df[1].values)}
            except Exception as e:
                print(f"[WARNING] Could not read .bim file: {e}. Falling back to an identity matrix.")
                self.bfile_path = None

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

        # Fallback: no reference panel, so SNPs are treated as independent.
        if self.bfile_path is None or self._snp_map is None:
            return np.eye(n_snps)

        # Map SNPs to reference-panel indices.
        snp_list = block_df['SNP'].values
        found_indices, found_mask = self._get_snp_indices(snp_list)

        # Initialize as identity; unmatched SNPs remain independent.
        L = np.eye(n_snps)

        if len(found_indices) == 0:
            return L

        # Read genotype matrix.
        try:
            from bed_reader import open_bed
            with open_bed(self.bfile_path) as bed:
                G = bed.read(index=found_indices)  # [N_individuals, M_found_snps]
        except Exception as e:
            print(f"[WARNING] Could not read .bed file: {e}. Falling back to an identity matrix.")
            return L

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

        return L
