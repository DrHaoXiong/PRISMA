import argparse
import numpy as np
import pandas as pd
import os
import sys
import time
import platform
from pathlib import Path
try:
    import resource
except ImportError:
    resource = None

# Module imports
from loader import TensorDataLoader
from partition import GenomicPartitioner
from builder import TensorBuilder
from solver import CoupledTensorSolver
from tune_rank import select_rank
from qc import initialize_qc_report, add_tensor_and_ld_qc, print_qc_summary, resolve_input_path

IDENTITY_LD_MESSAGE = (
    "Identity Laplacian was requested or implied without an empirical LD reference. "
    "For real-data analyses, provide --bfile with --ld-reference-mode plink/auto. "
    "To intentionally run diagnostic identity-LD mode, rerun with "
    "--ld-reference-mode identity --allow-identity-ld."
)


def manifest_allows_auto_identity(manifest_path):
    """Allow auto identity-LD only for bundled examples or tests."""
    parts = {part.lower() for part in Path(manifest_path).resolve().parts}
    return "examples" in parts or "tests" in parts


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

def main():
    parser = argparse.ArgumentParser(description="Run the PRISMA core decomposition workflow.")
    parser.add_argument("--manifest", required=True, help="Input data manifest CSV.")
    parser.add_argument("--out", default="./results", help="Output directory.")
    parser.add_argument("--rank", default="3", help="Decomposition rank integer, 0, or 'auto'.")
    parser.add_argument("--max-rank", "--max_tune_rank", dest="max_rank", type=int, default=5, help="Maximum rank to scan during automatic rank selection.")
    parser.add_argument("--corcondia-threshold", type=float, default=80.0, help="CORCONDIA threshold for automatic rank selection.")
    parser.add_argument("--rank-seed", type=int, default=0, help="Random seed used during rank selection.")
    parser.add_argument("--auto_rank", action="store_true", help="Force automatic rank selection.")
    parser.add_argument("--iter", type=int, default=20, help="Maximum number of ALS iterations.")
    parser.add_argument("--sample_test", action="store_true", help="Run a quick test using a 50,000-SNP random subset.")
    parser.add_argument("--benchmark_log", default=None, help="Optional benchmark CSV file to append runtime statistics.")
    parser.add_argument("--benchmark_run_id", default=None, help="Optional benchmark run identifier.")
    parser.add_argument("--no_banner", action="store_true", help="Suppress the PRISMA startup banner.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for solver initialization. Use a negative value to leave it unset.")
    parser.add_argument("--phenotype-name", default=None, help="Phenotype name written to Factor_C_Phenotypes.csv.")
    parser.add_argument("--bfile", default=None, help="PLINK binary reference prefix, expecting .bed/.bim/.fam.")
    parser.add_argument("--ld-reference-mode", choices=["plink", "identity", "auto"], default="auto", help="LD reference mode.")
    parser.add_argument("--allow-identity-ld", action="store_true", help="Allow identity Laplacian mode for synthetic or diagnostic runs.")
    parser.add_argument("--quiet-blocks", action="store_true", help="Suppress per-block logs while preserving summary QC.")
    parser.add_argument("--ld-min-overlap", type=int, default=2, help="Minimum SNP overlap per LD block for empirical LD construction.")
    parser.add_argument("--ld-coverage-warning", type=float, default=0.80, help="Warning threshold for LD reference SNP coverage.")
    parser.add_argument("--ld-coverage-fail", type=float, default=0.50, help="Fail threshold for LD reference SNP coverage.")
    parser.add_argument("--allow-low-coverage", action="store_true", help="Continue despite low LD reference coverage.")
    parser.add_argument("--allele-match-warning", type=float, default=0.90, help="Warning threshold for allele match rate.")
    parser.add_argument("--allele-match-fail", type=float, default=0.70, help="Fail threshold for allele match rate.")
    parser.add_argument("--allow-low-allele-match", action="store_true", help="Continue despite low allele match rate.")
    parser.add_argument("--tissue-nonzero-warning", type=float, default=0.01, help="Warning threshold for tissue nonzero rate.")
    parser.add_argument("--tissue-nonzero-fail", type=float, default=0.001, help="Fail threshold for tissue nonzero rate.")
    parser.add_argument("--allow-low-tissue-nonzero", action="store_true", help="Continue despite low tissue nonzero rate.")
    args = parser.parse_args()

    if not args.no_banner:
        print_project_banner()

    if args.seed >= 0:
        np.random.seed(args.seed)

    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    os.makedirs(args.out, exist_ok=True)
    if args.quiet_blocks:
        os.environ['PRISMA_QUIET_BLOCKS'] = '1'

    # Resolve LD-reference mode early.
    identity_ld_active = False
    identity_ld_reason = None
    if args.ld_reference_mode == "plink" and not args.bfile:
        print("[ERROR] --ld-reference-mode plink requires --bfile.")
        sys.exit(1)
    if args.ld_reference_mode == "identity" and not args.allow_identity_ld:
        print(f"[ERROR] {IDENTITY_LD_MESSAGE}")
        sys.exit(1)
    bfile_path = args.bfile if args.ld_reference_mode in {"plink", "auto"} and args.bfile else None
    if bfile_path is None:
        if args.ld_reference_mode == "auto":
            if args.allow_identity_ld:
                identity_ld_active = True
                identity_ld_reason = "explicit_allow_identity_ld"
                print(f"[WARNING] {IDENTITY_LD_MESSAGE}")
            elif manifest_allows_auto_identity(args.manifest):
                identity_ld_active = True
                identity_ld_reason = "examples_or_tests_manifest"
                print(
                    f"[WARNING] {IDENTITY_LD_MESSAGE} "
                    "Proceeding because the manifest is under examples/ or tests/."
                )
            else:
                print(f"[ERROR] {IDENTITY_LD_MESSAGE}")
                sys.exit(1)
        elif args.ld_reference_mode == "identity":
            identity_ld_active = True
            identity_ld_reason = "explicit_identity_mode"
            print(f"[WARNING] {IDENTITY_LD_MESSAGE}")
    else:
        missing_plink = [f"{bfile_path}{suffix}" for suffix in [".bed", ".bim", ".fam"] if not os.path.exists(f"{bfile_path}{suffix}")]
        if missing_plink:
            print(f"[ERROR] Missing PLINK reference files: {missing_plink}")
            sys.exit(1)

    # 1. Load data.
    manifest_df = pd.read_csv(args.manifest)
    bed_rows = manifest_df[manifest_df['type'] == 'bed']
    if len(bed_rows) == 0:
        print("[ERROR] Manifest must contain a row with type='bed' for LD block definitions.")
        sys.exit(1)
    bed_path = resolve_input_path(bed_rows.iloc[0]['path'], args.manifest)
    gwas_rows = manifest_df[manifest_df['type'] == 'gwas']
    if args.phenotype_name:
        phenotype_name = args.phenotype_name
    elif len(gwas_rows) > 0 and pd.notna(gwas_rows.iloc[0]['name']):
        phenotype_name = str(gwas_rows.iloc[0]['name'])
    else:
        phenotype_name = "trait"

    try:
        qc_report = initialize_qc_report(
            args.manifest,
            args.out,
            allele_match_warning=args.allele_match_warning,
            allele_match_fail=args.allele_match_fail,
            allow_low_allele_match=args.allow_low_allele_match,
        )
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
    print(f"   - Phenotype name: {phenotype_name}")
    print(f"   - GWAS: {gwas_z_col}")
    print(f"   - Tissues: {[c.replace('_Z', '') for c in tissue_cols]}")

    # 3. Partition into LD blocks.
    print("[INFO] Partitioning variants into LD blocks...")
    partitioner = GenomicPartitioner(df)
    block_defs = partitioner.load_block_definitions(bed_path)

    # 4. Initialize tensor builder.
    builder = TensorBuilder(
        n_tissues,
        n_phenos,
        bfile_path=bfile_path,
        ld_reference_mode=args.ld_reference_mode,
        ld_min_overlap=args.ld_min_overlap,
        quiet_blocks=args.quiet_blocks,
    )
    builder.gwas_z_col = gwas_z_col
    builder.tissue_cols = tissue_cols

    try:
        qc_report["run_configuration"] = {
            "phenotype_name": phenotype_name,
            "ld_reference_mode": args.ld_reference_mode,
            "bfile": bfile_path,
            "allow_identity_ld": bool(args.allow_identity_ld),
            "identity_ld_active": bool(identity_ld_active),
            "identity_ld_reason": identity_ld_reason,
        }
        if identity_ld_active:
            qc_report.setdefault("warnings", []).append(IDENTITY_LD_MESSAGE)
        qc_report = add_tensor_and_ld_qc(
            qc_report,
            df,
            tissue_cols,
            block_defs,
            partitioner,
            builder,
            args.out,
            tissue_nonzero_warning=args.tissue_nonzero_warning,
            tissue_nonzero_fail=args.tissue_nonzero_fail,
            allow_low_tissue_nonzero=args.allow_low_tissue_nonzero,
            ld_coverage_warning=args.ld_coverage_warning,
            ld_coverage_fail=args.ld_coverage_fail,
            allow_low_coverage=args.allow_low_coverage,
        )
        print_qc_summary(qc_report)
    except Exception as e:
        print(f"[ERROR] QC failed before factorization: {e}")
        sys.exit(1)

    # 5. Determine rank.
    rank_value = str(args.rank).strip().lower()
    if args.auto_rank or rank_value in {"auto", "0"}:
        final_rank, _, selection = select_rank(
            partitioner,
            builder,
            block_defs,
            n_tissues,
            n_phenos,
            max_rank=args.max_rank,
            corcondia_threshold=args.corcondia_threshold,
            rank_seed=args.rank_seed,
            max_iter=5,
            out_dir=args.out,
        )
        print(f"[INFO] Automatic rank selection complete. Selected Rank={final_rank} ({selection['selection_rule']}).")
    else:
        try:
            final_rank = int(rank_value)
        except ValueError:
            print("[ERROR] --rank must be an integer, 0, or 'auto'.")
            sys.exit(1)
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
    a_df = pd.DataFrame(A, index=all_snps, columns=[f'Rank_{i}' for i in range(final_rank)])
    a_df.to_csv(os.path.join(args.out, 'Factor_A_SNPs.csv'))

    b_df = pd.DataFrame(B, index=[c.replace('_Z', '') for c in tissue_cols],
                        columns=[f'Rank_{i}' for i in range(final_rank)])
    b_df.to_csv(os.path.join(args.out, 'Factor_B_Tissues.csv'))

    c_df = pd.DataFrame(C, index=[phenotype_name],
                        columns=[f'Rank_{i}' for i in range(final_rank)])
    c_df.to_csv(os.path.join(args.out, 'Factor_C_Phenotypes.csv'))

    qc_report["postfit_laplacian_usage"] = builder.summarize_laplacian_usage()
    try:
        from qc import write_qc_reports
        write_qc_reports(qc_report, args.out)
    except Exception as e:
        print(f"[WARNING] Could not update postfit QC usage report: {e}")

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
