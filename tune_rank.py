import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
import json
from scipy.linalg import khatri_rao

from loader import TensorDataLoader
from partition import GenomicPartitioner
from builder import TensorBuilder
from solver import CoupledTensorSolver
from qc import resolve_input_path

def print_project_banner():
    banner = """
========================================================================
PRISMA
Polygenic Risk Integration via Summary-statistics Multi-tissue
Array-decomposition

Contact: dr.haoxiong15@gmail.com
License: Apache-2.0

Please cite the accompanying PRISMA manuscript/preprint when available.

Rank tuning utility for PRISMA. PRISMA is research software and is not
intended for clinical diagnosis or treatment decisions.
========================================================================
"""
    print(banner.strip())

def calculate_fit(partitioner, builder, block_defs, B, C, local_As):
    """
    Compute model variance explained.
    Fit = 1 - (||X - Model||^2 / ||X||^2)
    """
    total_error_sq = 0
    total_var_sq = 0

    block_idx = 0
    for block_id, block_df in partitioner.iter_blocks(block_defs):
        X_i = builder.build_tensor(block_df)
        n_snps = X_i.shape[0]

        A_i = local_As[block_idx]
        KB = khatri_rao(B, C)

        X_model_unfold = A_i @ KB.T
        X_real_unfold = X_i.reshape(n_snps, -1)

        total_error_sq += np.sum((X_real_unfold - X_model_unfold) ** 2)
        total_var_sq += np.sum(X_real_unfold ** 2)

        block_idx += 1

    fit = 1 - (total_error_sq / total_var_sq)
    return fit

def compute_corcondia(X, A, B, C):
    """
    Compute CORCONDIA (Core Consistency Diagnostic).
    CORCONDIA = 100 * (1 - ||G_opt - G_super||^2 / ||G_super||^2)

    For a single phenotype (P=1), C is a (1, R) vector and CP decomposition
    degenerates into matrix factorization, so the diagnostic is adjusted.
    """
    rank = A.shape[1]
    N, T, P = X.shape

    # Use a simplified diagnostic for the single-phenotype case (P=1).
    if P == 1:
        # X degenerates into a matrix (N, T).
        X_mat = X.reshape(N, T)

        # Reconstruct matrix: X ~= A * diag(C) * B^T.
        C_diag = np.diag(C.flatten())  # (R, R)
        X_recon = A @ C_diag @ B.T  # (N, T)

        # Standard CORCONDIA is not directly applicable for P=1; use relative error.
        error = np.linalg.norm(X_mat - X_recon, 'fro')
        total = np.linalg.norm(X_mat, 'fro')

        # Convert to a CORCONDIA-like percentage.
        corcondia = 100 * (1 - (error / (total + 1e-12))**2)
        return corcondia

    # Standard CORCONDIA for multiple phenotypes (P > 1).
    # Mode-1 unfolding
    X_unfold = X.reshape(N, -1)  # (N, T*P)

    # Khatri-Rao product
    V = khatri_rao(C, B)  # (T*P, R)

    # Compute pseudoinverses
    A_pinv = np.linalg.pinv(A)  # (R, N)
    V_pinv = np.linalg.pinv(V)  # (R, T*P)

    # Extract core tensor
    G_unfold = A_pinv @ X_unfold @ V_pinv.T  # (R, R*R) for P>1

    # Reshape to 3D core tensor
    G_opt = G_unfold.reshape(rank, rank, rank)

    # Superdiagonal tensor (identity)
    G_super = np.zeros((rank, rank, rank))
    for r in range(rank):
        G_super[r, r, r] = 1.0

    # Compute CORCONDIA
    diff_norm = np.linalg.norm(G_opt - G_super)
    super_norm = np.linalg.norm(G_super)

    corcondia = 100 * (1 - (diff_norm ** 2) / (super_norm ** 2 + 1e-12))
    return corcondia


def _elbow_rank(ranks, fits):
    if len(ranks) < 3:
        return ranks[int(np.argmax(fits))]
    diffs = np.diff(fits)
    second_diffs = np.diff(diffs)
    if len(second_diffs) == 0:
        return ranks[int(np.argmax(fits))]
    # The largest negative change in slope marks a simple variance-explained elbow.
    return ranks[int(np.argmin(second_diffs) + 1)]


def select_rank(
    partitioner,
    builder,
    block_defs,
    n_tissues,
    n_phenos,
    max_rank=5,
    corcondia_threshold=80.0,
    rank_seed=0,
    max_iter=5,
    out_dir=None,
):
    """
    Shared PRISMA rank-selection routine used by run_prisma.py and tune_rank.py.
    """
    if rank_seed >= 0:
        np.random.seed(rank_seed)

    ranks = list(range(1, int(max_rank) + 1))
    fits = []
    concordias = []

    for r in ranks:
        print(f"\n[INFO] Evaluating Rank = {r} ...")
        solver = CoupledTensorSolver(n_tissues, n_phenos, rank=r, max_iter=max_iter)
        B, C = solver.train(partitioner, builder, block_defs)

        local_As = []
        corcondia_scores = []
        for _, block_df in partitioner.iter_blocks(block_defs):
            X_i = builder.build_tensor(block_df)
            L_i = builder.build_laplacian(block_df)
            A_i, _ = solver.solve_local_A(X_i, L_i)
            local_As.append(A_i)
            corcondia_scores.append(compute_corcondia(X_i, A_i, B, C))

        fit = calculate_fit(partitioner, builder, block_defs, B, C, local_As)
        avg_corcondia = float(np.mean(corcondia_scores)) if corcondia_scores else float("nan")
        fits.append(float(fit))
        concordias.append(avg_corcondia)
        print(f"   Rank {r} -> Fit: {fit:.4f}, CORCONDIA: {avg_corcondia:.2f}%")

    passing = [
        idx for idx, (rank_value, corcondia) in enumerate(zip(ranks, concordias))
        if rank_value > 1 and corcondia >= corcondia_threshold
    ]
    if passing:
        selected_idx = passing[0]
        selection_rule = (
            f"selected lowest non-trivial rank with CORCONDIA >= {corcondia_threshold:.1f}%"
        )
        fallback_used = False
    else:
        selected_rank_fallback = _elbow_rank(ranks, fits)
        selected_idx = ranks.index(selected_rank_fallback)
        selection_rule = "variance-explained elbow fallback because no rank passed CORCONDIA threshold"
        fallback_used = True

    selected_rank = int(ranks[selected_idx])
    diagnostics = pd.DataFrame({
        "Rank": ranks,
        "Variance_Explained": fits,
        "CORCONDIA": concordias,
    })
    selection = {
        "candidate_ranks": ranks,
        "selected_rank": selected_rank,
        "selection_rule": selection_rule,
        "corcondia_threshold": float(corcondia_threshold),
        "corcondia_values": concordias,
        "fit_values": fits,
        "fallback_used": fallback_used,
    }

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        diagnostics.to_csv(os.path.join(out_dir, "rank_diagnostics.csv"), index=False)
        with open(os.path.join(out_dir, "rank_selection.json"), "w", encoding="utf-8") as handle:
            json.dump(selection, handle, indent=2, sort_keys=True)

    return selected_rank, diagnostics, selection

def run_tuning(manifest_path, bed_path, max_rank=10, seed=42, corcondia_threshold=80.0, out_dir="results"):
    """
    Main automatic rank-tuning workflow.
    """
    if seed >= 0:
        np.random.seed(seed)

    print(f"[INFO] Starting automatic rank tuning (Rank 1 - {max_rank})...")

    # 1. Load data.
    print("[INFO] Loading input data...")
    loader = TensorDataLoader(manifest_path, apply_genomic_control=True)
    df = loader.load_and_align()

    # 2. Infer dimensions.
    gwas_z_col = 'GWAS_Z'
    tissue_cols = [c for c in df.columns if c.endswith('_Z') and c != 'GWAS_Z']

    n_phenos = 1
    n_tissues = len(tissue_cols)

    print(f"[INFO] Detected dimensions: Tissues={n_tissues}, Phenotypes={n_phenos}")

    # 3. Initialize partitioner and builder.
    partitioner = GenomicPartitioner(df)
    block_defs = partitioner.load_block_definitions(bed_path)

    builder = TensorBuilder(n_tissues, n_phenos)
    builder.gwas_z_col = gwas_z_col
    builder.tissue_cols = tissue_cols

    selected_rank, results_df, selection = select_rank(
        partitioner,
        builder,
        block_defs,
        n_tissues,
        n_phenos,
        max_rank=max_rank,
        corcondia_threshold=corcondia_threshold,
        rank_seed=seed,
        max_iter=5,
        out_dir=out_dir,
    )
    ranks = list(results_df["Rank"])
    fits = list(results_df["Variance_Explained"])
    concordias = list(results_df["CORCONDIA"])
    selected_idx = ranks.index(selected_rank)
    selection_note = selection["selection_rule"]

    # 5. Plot rank-selection diagnostics.
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'sans-serif'

    # Publication-friendly blue palette.
    COLOR_PRIMARY = '#08519c'
    COLOR_SECONDARY = '#3182bd'
    COLOR_THRESHOLD = '#6baed6'
    COLOR_SELECTED = '#e74c3c'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # === Left panel: Variance Explained (Scree Plot) ===
    ax1.plot(ranks, fits, 'o-', linewidth=2.5, markersize=7,
             color=COLOR_PRIMARY, label='Variance Explained', zorder=2)

    # Highlight the dynamically selected rank.
    ax1.plot(selected_rank, fits[selected_idx], 'o', markersize=10,
             color=COLOR_SELECTED, zorder=10,
             markeredgewidth=2, markeredgecolor=COLOR_SELECTED,
             markerfacecolor='white', label=f'Selected (R={selected_rank})')

    ax1.set_title('A. Variance Explained', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Rank (R)', fontsize=10)
    ax1.set_ylabel('Variance Explained', fontsize=10)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax1.set_xticks(ranks)
    ax1.legend(loc='lower right', fontsize=8, frameon=True)

    # === Right panel: CORCONDIA Diagnostic ===
    ax2.plot(ranks, concordias, 's-', linewidth=2.5, markersize=7,
             color=COLOR_SECONDARY, label='CORCONDIA', zorder=2)

    # Highlight the dynamically selected rank.
    ax2.plot(selected_rank, concordias[selected_idx], 's', markersize=10,
             color=COLOR_SELECTED, zorder=10,
             markeredgewidth=2, markeredgecolor=COLOR_SELECTED,
             markerfacecolor='white', label=f'Selected (R={selected_rank})')

    # Diagnostic threshold.
    ax2.axhline(y=80, color=COLOR_THRESHOLD, linestyle='--', linewidth=2,
                label='Threshold (80%)', zorder=0)

    # Annotate selected rank.
    ax2.text(selected_rank, concordias[selected_idx] + 5,
            f'R={selected_rank}: {concordias[selected_idx]:.1f}%',
            fontsize=8, ha='center', color=COLOR_SELECTED, fontweight='normal')

    ax2.set_title('B. CORCONDIA Diagnostic', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Rank (R)', fontsize=10)
    ax2.set_ylabel('CORCONDIA (%)', fontsize=10)
    ax2.set_ylim([0, 105])
    ax2.legend(loc='lower right', fontsize=8, frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax2.set_xticks(ranks)

    # Overall title.
    fig.suptitle(f'Rank Selection: R={selected_rank} {selection_note}',
                 fontsize=11, fontweight='bold', y=1.00)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'rank_selection.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'rank_selection.pdf'), dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Figure saved: {os.path.join(out_dir, 'rank_selection.png')}")
    print(f"[INFO] Figure saved: {os.path.join(out_dir, 'rank_selection.pdf')}")

    # 6. Save rank-selection data.
    results_df.to_csv(os.path.join(out_dir, 'Rank_Selection_Results.csv'), index=False)
    print(f"[INFO] Data saved: {os.path.join(out_dir, 'Rank_Selection_Results.csv')}")
    print("\nRank Selection Results:")
    print(results_df.to_string(index=False))
    print(f"\n[INFO] Selected Rank: R={selected_rank} ({selection_note}).")
    print("\n[INFO] Suggested criterion: choose a rank with CORCONDIA > 80% "
          "where variance explained starts to plateau.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISMA automatic rank tuning.")
    parser.add_argument("--manifest", required=True, help="Input data manifest CSV.")
    parser.add_argument("--bed", required=True, help="LD block BED file.")
    parser.add_argument("--max_rank", type=int, default=10, help="Maximum rank to evaluate.")
    parser.add_argument("--corcondia-threshold", type=float, default=80.0, help="CORCONDIA threshold for automatic rank selection.")
    parser.add_argument("--out", default="results", help="Output directory for rank diagnostics.")
    parser.add_argument("--no_banner", action="store_true", help="Suppress the PRISMA startup banner.")
    parser.add_argument("--seed", "--rank-seed", dest="seed", type=int, default=42, help="Random seed for rank selection. Use a negative value to leave it unset.")
    args = parser.parse_args()

    if not args.no_banner:
        print_project_banner()

    manifest_df = pd.read_csv(args.manifest)
    bed_rows = manifest_df[manifest_df['type'] == 'bed']
    bed_path = bed_rows.iloc[0]['path'] if len(bed_rows) > 0 else args.bed
    bed_path = resolve_input_path(bed_path, args.manifest)

    run_tuning(args.manifest, bed_path, args.max_rank, args.seed, args.corcondia_threshold, args.out)
