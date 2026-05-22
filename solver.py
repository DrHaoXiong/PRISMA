import pandas as pd
import numpy as np
from scipy.linalg import solve, khatri_rao
from scipy.sparse.linalg import cg, LinearOperator

class CoupledTensorSolver:
    """
    PRISMA core solver.

    Algorithm: block-wise alternating least squares (ALS) with graph regularization.

    Objective:
    min sum_i ( || X_i - [[A_i, B, C]] ||^2 + lambda * Tr(A_i.T @ L_i @ A_i) )

    X_i: block-level data tensor [SNPs, Tissues, Phenotypes]
    A_i: local SNP factor matrix [SNPs, Rank]
    B:   global tissue factor matrix [Tissues, Rank]
    C:   global phenotype factor matrix [Phenotypes, Rank]
    L_i: local graph Laplacian [SNPs, SNPs]
    """

    def __init__(self, n_tissues, n_phenos, rank=3, lambda_reg=0.1, lambda_b=0.1, lambda_c=0.1,
                 warmup_ratio=0.3, max_iter=10, tol=1e-4):
        self.rank = rank
        self.lambda_reg = lambda_reg
        self.lambda_b = lambda_b
        self.lambda_c = lambda_c
        self.warmup_ratio = warmup_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.n_tissues = n_tissues
        self.n_phenos = n_phenos

        # Randomly initialize global factors. The caller controls the random seed.
        self.B = np.abs(np.random.randn(n_tissues, rank))
        self.C = np.abs(np.random.randn(n_phenos, rank))

        # Normalize initial factors.
        self.B /= np.linalg.norm(self.B, axis=0)
        self.C /= np.linalg.norm(self.C, axis=0)

    def solve_local_A(self, tensor_block, laplacian):
        """
        Step 1: update the local SNP factor matrix A_i.
        """
        n_snps = tensor_block.shape[0]
        X_unfold = tensor_block.reshape(n_snps, -1)
        KB = khatri_rao(self.B, self.C)
        RHS = X_unfold @ KB

        CTC = self.C.T @ self.C
        BTB = self.B.T @ self.B
        M = CTC * BTB

        A_new = np.zeros((n_snps, self.rank))

        for r in range(self.rank):
            m_rr = M[r, r]
            rhs_r = RHS[:, r]

            def mv(v):
                return m_rr * v + self.lambda_reg * (laplacian @ v)

            A_op = LinearOperator((n_snps, n_snps), matvec=mv)
            col_sol, _ = cg(A_op, rhs_r, rtol=1e-5)
            A_new[:, r] = col_sol

        return A_new, KB

    def train(self, partitioner, builder, block_defs):
        """
        Main training loop with warmup and L2 regularization.
        """
        print(f"[INFO] Starting training (Rank={self.rank}, Lambda={self.lambda_reg})")

        # Warmup period before applying the non-negativity constraint.
        warmup_epochs = int(self.max_iter * self.warmup_ratio)
        print(f"       Warmup: negative values are allowed for the first {warmup_epochs} epochs.")

        for epoch in range(self.max_iter):
            B_num = np.zeros_like(self.B)
            B_denom = np.zeros((self.rank, self.rank))
            C_num = np.zeros_like(self.C)
            C_denom = np.zeros((self.rank, self.rank))

            n_blocks = 0

            for block_id, block_df in partitioner.iter_blocks(block_defs):
                n_blocks += 1

                X_i = builder.build_tensor(block_df)
                L_i = builder.build_laplacian(block_df)
                A_i, KB = self.solve_local_A(X_i, L_i)

                for p in range(self.n_phenos):
                    X_p = X_i[:, :, p]
                    AC = A_i * self.C[p, :]
                    B_num += X_p.T @ AC

                ATA = A_i.T @ A_i
                CTC = self.C.T @ self.C
                B_denom += ATA * CTC

                for t in range(self.n_tissues):
                    X_t = X_i[:, t, :]
                    AB = A_i * self.B[t, :]
                    C_num += X_t.T @ AB

                BTB = self.B.T @ self.B
                C_denom += ATA * BTB

            # Global updates with L2 regularization.
            reg_B = self.lambda_b * np.eye(self.rank)
            reg_C = self.lambda_c * np.eye(self.rank)

            self.B = solve(B_denom + reg_B, B_num.T).T
            self.C = solve(C_denom + reg_C, C_num.T).T

            # Delayed non-negativity constraint after warmup.
            if epoch >= warmup_epochs:
                self.B[self.B < 0] = 1e-4
                self.C[self.C < 0] = 1e-4

            # Safe normalization.
            norm_B = np.linalg.norm(self.B, axis=0)
            norm_B[norm_B < 1e-12] = 1.0
            self.B /= norm_B

            norm_C = np.linalg.norm(self.C, axis=0)
            norm_C[norm_C < 1e-12] = 1.0
            self.C /= norm_C

            print(f"Epoch {epoch+1}/{self.max_iter} complete. Processed {n_blocks} blocks.")

        print("[INFO] Training complete.")
        return self.B, self.C
