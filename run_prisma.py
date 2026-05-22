import argparse
import numpy as np
import pandas as pd
import os
import sys
import time
import platform
try:
    import resource
except ImportError:
    resource = None

# Module imports
from loader import TensorDataLoader
from partition import GenomicPartitioner
from builder import TensorBuilder
from solver import CoupledTensorSolver
from tune_rank import compute_corcondia

def print_project_banner():
    banner = """
========================================================================
PRISMA
Polygenic Risk Integration via Summary-statistics Multi-tissue
Array-decomposition

Contact: dr.haoxiong15@gmail.com
License: Apache-2.0

Please cite the accompanying PRISMA manuscript/preprint when available.

PRISMA is research software for decomposing GWAS summary statistics and
multi-tissue eQTL evidence into interpretable tissue-anchored polygenic
axes. It is not intended for clinical diagnosis or treatment decisions.
========================================================================
"""
    print(banner.strip())

def auto_select_rank(partitioner, builder, block_defs, n_tissues, n_phenos, max_rank=8):
    """
    Automatically select the decomposition rank by scanning Rank 1 to max_rank.
    """
    print(f"\n[INFO] Entering automatic rank selection mode (Rank 1-{max_rank})...")

    concordias = []
    ranks = range(1, max_rank + 1)

    for r in ranks:
        print(f"   Evaluating Rank={r}...", end=" ")

        solver = CoupledTensorSolver(n_tissues, n_phenos, rank=r, max_iter=5)
        B, C = solver.train(partitioner, builder, block_defs)

        corcondia_scores = []
        for _, block_df in partitioner.iter_blocks(block_defs):
            X_i = builder.build_tensor(block_df)
            L_i = builder.build_laplacian(block_df)
            A_i, _ = solver.solve_local_A(X_i, L_i)

            corcondia = compute_corcondia(X_i, A_i, B, C)
            corcondia_scores.append(corcondia)

        avg_corcondia = np.mean(corcondia_scores)
        concordias.append(avg_corcondia)
        print(f"CORCONDIA={avg_corcondia:.2f}%")

    # Selection strategy: avoid the trivial Rank=1 solution.
    valid_ranks = [r for r, c in zip(ranks, concordias) if c > 80 and r > 1]
    if valid_ranks:
        best_rank = max(valid_ranks)
        print("\n[INFO] Strategy: choose the largest Rank with CORCONDIA > 80%, excluding Rank=1.")
    else:
        if len(concordias) >= 3:
            diffs = np.diff(concordias)
            if len(diffs) > 1:
                second_diffs = np.diff(diffs)
                elbow_idx = np.argmin(second_diffs) + 1
                best_rank = max(ranks[elbow_idx], 3)
                print("\n[INFO] Strategy: choose the elbow Rank with a minimum of Rank=3.")
            else:
                best_rank = 3
                print("\n[INFO] Strategy: conservatively choose Rank=3.")
        else:
            best_rank = 3
            print("\n[INFO] Strategy: conservatively choose Rank=3.")

    print(f"[INFO] Automatic rank selection complete. Selected Rank={best_rank} "
          f"(CORCONDIA={concordias[best_rank-1]:.2f}%).")
    return best_rank

def main():
    parser = argparse.ArgumentParser(description="Run the PRISMA core decomposition workflow.")
    parser.add_argument("--manifest", required=True, help="Input data manifest CSV.")
    parser.add_argument("--out", default="./results", help="Output directory.")
    parser.add_argument("--rank", type=int, default=0, help="Decomposition rank. Use 0 for automatic rank selection.")
    parser.add_argument("--max_tune_rank", type=int, default=8, help="Maximum rank to scan during automatic rank selection.")
    parser.add_argument("--auto_rank", action="store_true", help="Force automatic rank selection.")
    parser.add_argument("--iter", type=int, default=20, help="Maximum number of ALS iterations.")
    parser.add_argument("--sample_test", action="store_true", help="Run a quick test using a 50,000-SNP random subset.")
    parser.add_argument("--benchmark_log", default=None, help="Optional benchmark CSV file to append runtime statistics.")
    parser.add_argument("--benchmark_run_id", default=None, help="Optional benchmark run identifier.")
    parser.add_argument("--no_banner", action="store_true", help="Suppress the PRISMA startup banner.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for solver initialization. Use a negative value to leave it unset.")
    args = parser.parse_args()

    if not args.no_banner:
        print_project_banner()

    if args.seed >= 0:
        np.random.seed(args.seed)

    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    # 1. Load data.
    manifest_df = pd.read_csv(args.manifest)
    bed_rows = manifest_df[manifest_df['type'] == 'bed']
    if len(bed_rows) == 0:
        print("[ERROR] Manifest must contain a row with type='bed' for LD block definitions.")
        sys.exit(1)
    bed_path = bed_rows.iloc[0]['path']

    try:
        loader = TensorDataLoader(args.manifest, apply_genomic_control=True)
        df = loader.load_and_align()
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        sys.exit(1)

    # 1.5 Quick test mode.
    if args.sample_test:
        n_sample = min(50000, len(df))
        print(f"[INFO] Test mode: randomly sampling {n_sample} SNPs...")
        df = df.sample(n=n_sample, random_state=args.seed if args.seed >= 0 else None).sort_values(['CHR', 'BP'])

    # 2. Infer data dimensions.
    gwas_z_col = 'GWAS_Z'
    tissue_cols = [c for c in df.columns if c.endswith('_Z') and c != 'GWAS_Z']

    n_phenos = 1
    n_tissues = len(tissue_cols)

    print(f"[INFO] Detected dimensions: Phenotypes={n_phenos}, Tissues={n_tissues}")
    print(f"   - GWAS: {gwas_z_col}")
    print(f"   - Tissues: {[c.replace('_Z', '') for c in tissue_cols]}")

    # 3. Partition into LD blocks.
    print("[INFO] Partitioning variants into LD blocks...")
    partitioner = GenomicPartitioner(df)
    block_defs = partitioner.load_block_definitions(bed_path)

    # 4. Initialize tensor builder.
    builder = TensorBuilder(n_tissues, n_phenos)
    builder.gwas_z_col = gwas_z_col
    builder.tissue_cols = tissue_cols

    # 5. Determine rank.
    if args.auto_rank or args.rank <= 0:
        final_rank = auto_select_rank(partitioner, builder, block_defs, n_tissues, n_phenos, args.max_tune_rank)
    else:
        final_rank = args.rank
        print(f"[INFO] Using user-specified Rank: {final_rank}")

    # 6. Final training.
    print(f"\n[INFO] Starting final training (Rank={final_rank}, Iter={args.iter})...")
    solver = CoupledTensorSolver(n_tissues, n_phenos, rank=final_rank, max_iter=args.iter)
    B, C = solver.train(partitioner, builder, block_defs)

    # 7. Collect SNP factor matrices.
    print("[INFO] Collecting SNP factor matrices...")
    all_snps = []
    all_A = []

    for block_id, block_df in partitioner.iter_blocks(block_defs):
        X_i = builder.build_tensor(block_df)
        L_i = builder.build_laplacian(block_df)
        A_i, _ = solver.solve_local_A(X_i, L_i)

        all_snps.extend(block_df['SNP'].values)
        all_A.append(A_i)

    A = np.vstack(all_A)
    print(f"[INFO] Collection complete: {A.shape[0]} SNPs.")

    # 8. Save outputs.
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    a_df = pd.DataFrame(A, index=all_snps, columns=[f'Rank_{i}' for i in range(final_rank)])
    a_df.to_csv(os.path.join(args.out, 'Factor_A_SNPs.csv'))

    b_df = pd.DataFrame(B, index=[c.replace('_Z', '') for c in tissue_cols],
                        columns=[f'Rank_{i}' for i in range(final_rank)])
    b_df.to_csv(os.path.join(args.out, 'Factor_B_Tissues.csv'))

    c_df = pd.DataFrame(C, index=['DR'],
                        columns=[f'Rank_{i}' for i in range(final_rank)])
    c_df.to_csv(os.path.join(args.out, 'Factor_C_Phenotypes.csv'))

    print("[INFO] PRISMA run complete.")

    if args.benchmark_log:
        wall_seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start
        if resource is not None:
            peak_memory_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
        else:
            peak_memory_gb = np.nan
        benchmark_row = {
            'Run_ID': args.benchmark_run_id or 'run',
            'N_Raw_Backbone_Variants': int(loader.stats.get('N_Raw_Backbone_Variants', len(df))),
            'N_Candidate_Variants_PreLD': int(loader.stats.get('N_Candidate_Variants_PreLD', len(df))),
            'N_Candidate_Variant_Tissue_Cells_PreLD': int(loader.stats.get('N_Candidate_Variants_PreLD', len(df)) * n_tissues),
            'N_Gene_Representatives_PreBlacklist': int(loader.stats.get('N_Gene_Representatives_PreBlacklist', len(df))),
            'N_Final_Variants': int(len(df)),
            'N_Output_SNP_Factor_Rows': int(A.shape[0]),
            'N_Tissues': int(n_tissues),
            'N_Final_Variant_Tissue_Cells': int(len(df) * n_tissues),
            'N_LD_Blocks': int(len(all_A)),
            'Rank': int(final_rank),
            'Lambda': float(solver.lambda_reg),
            'Max_Iterations': int(args.iter),
            'Wall_Time_Seconds': wall_seconds,
            'CPU_Time_Seconds': cpu_seconds,
            'Peak_Memory_GB': peak_memory_gb,
            'Hardware': platform.processor() or platform.machine(),
            'Software_Environment': f"Python {platform.python_version()} on {platform.platform()}",
            'Command': ' '.join(sys.argv),
        }
        benchmark_log = os.path.abspath(args.benchmark_log)
        os.makedirs(os.path.dirname(benchmark_log), exist_ok=True)
        pd.DataFrame([benchmark_row]).to_csv(
            benchmark_log,
            mode='a',
            index=False,
            header=not os.path.exists(benchmark_log)
        )
        print(f"[INFO] Benchmark record appended to: {benchmark_log}")

if __name__ == "__main__":
    main()
