import numpy as np
import pandas as pd
import json
import subprocess
import sys
from pathlib import Path

from builder import TensorBuilder
from partition import GenomicPartitioner
from tune_rank import select_rank


def _small_rank_inputs():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(30)],
        "CHR": [1] * 30,
        "BP": np.arange(30) * 100 + 100,
        "GWAS_Z": rng.normal(size=30),
        "blood_Z": rng.normal(size=30),
        "retina_Z": rng.normal(size=30),
    })
    blocks = pd.DataFrame({"chr": [1], "start": [0], "stop": [4000], "block_id": ["b1"]})
    partitioner = GenomicPartitioner(df)
    builder = TensorBuilder(2, 1, ld_reference_mode="identity")
    builder.tissue_cols = ["blood_Z", "retina_Z"]
    return partitioner, builder, blocks


def test_select_rank_is_deterministic_for_same_seed(tmp_path):
    p1, b1, blocks1 = _small_rank_inputs()
    r1, _, _ = select_rank(p1, b1, blocks1, 2, 1, max_rank=3, rank_seed=7, max_iter=1, out_dir=tmp_path / "a")
    p2, b2, blocks2 = _small_rank_inputs()
    r2, _, _ = select_rank(p2, b2, blocks2, 2, 1, max_rank=3, rank_seed=7, max_iter=1, out_dir=tmp_path / "b")
    assert r1 == r2


def test_run_prisma_auto_rank_matches_tune_rank_cli(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(repo / "examples" / "generate_mini_fixture_1000.py")], cwd=repo, check=True)

    run_out = tmp_path / "run_prisma_auto"
    tune_out = tmp_path / "tune_rank"
    subprocess.run(
        [
            sys.executable,
            str(repo / "run_prisma.py"),
            "--manifest",
            "examples/mini_fixture_1000/manifest.csv",
            "--out",
            str(run_out),
            "--rank",
            "auto",
            "--max-rank",
            "3",
            "--rank-seed",
            "11",
            "--phenotype-name",
            "synthetic_trait",
            "--ld-reference-mode",
            "identity",
            "--allow-identity-ld",
            "--iter",
            "1",
            "--no_banner",
            "--quiet-blocks",
        ],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / "tune_rank.py"),
            "--manifest",
            "examples/mini_fixture_1000/manifest.csv",
            "--bed",
            "examples/mini_fixture_1000/ld_blocks_header.bed",
            "--max_rank",
            "3",
            "--rank-seed",
            "11",
            "--out",
            str(tune_out),
            "--no_banner",
        ],
        cwd=repo,
        check=True,
    )

    run_selection = json.loads((run_out / "rank_selection.json").read_text(encoding="utf-8"))
    tune_selection = json.loads((tune_out / "rank_selection.json").read_text(encoding="utf-8"))
    assert run_selection["selected_rank"] == tune_selection["selected_rank"]
