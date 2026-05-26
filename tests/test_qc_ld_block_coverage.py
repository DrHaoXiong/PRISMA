import pandas as pd

from builder import TensorBuilder
from partition import GenomicPartitioner
from qc import _qc_ld_block_coverage


def test_ld_block_assignment_fraction():
    df = pd.DataFrame({
        "SNP": ["rs1", "rs2", "rs3"],
        "CHR": [1, 1, 2],
        "BP": [100, 200, 100],
    })
    blocks = pd.DataFrame({
        "chr": [1],
        "start": [0],
        "stop": [500],
        "block_id": ["b1"],
    })
    partitioner = GenomicPartitioner(df)
    builder = TensorBuilder(1, 1, ld_reference_mode="identity")
    report = _qc_ld_block_coverage(df, blocks, partitioner, builder)
    assert report["n_snps_assigned_to_block"] == 2
    assert report["n_snps_unassigned_to_block"] == 1
