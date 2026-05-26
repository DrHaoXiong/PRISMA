import numpy as np
import pandas as pd

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
