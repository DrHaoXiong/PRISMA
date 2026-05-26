import numpy as np
import pandas as pd
from bed_reader import to_bed

from builder import TensorBuilder


def test_builder_uses_empirical_plink_ld(tmp_path):
    prefix = tmp_path / "tiny_ref"
    genotypes = np.array([
        [0.0, 0.0, 2.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 0.0],
        [0.0, 1.0, 2.0],
    ])
    to_bed(
        str(prefix) + ".bed",
        genotypes,
        properties={
            "sid": ["rs1", "rs2", "rs3"],
            "chromosome": ["1", "1", "1"],
            "bp_position": [100, 200, 300],
            "allele_1": ["A", "A", "A"],
            "allele_2": ["C", "C", "C"],
        },
    )
    block = pd.DataFrame({"SNP": ["rs1", "rs2"], "CHR": [1, 1], "BP": [100, 200]})
    builder = TensorBuilder(1, 1, bfile_path=str(prefix), ld_reference_mode="plink", ld_min_overlap=2)
    laplacian = builder.build_laplacian(block)
    stats = builder.summarize_laplacian_usage()
    assert laplacian.shape == (2, 2)
    assert stats["n_blocks_empirical_laplacian"] == 1
    assert stats["fraction_snps_with_empirical_ld"] == 1.0
