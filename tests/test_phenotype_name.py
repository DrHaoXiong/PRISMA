import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_phenotype_name_written(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(repo / "examples" / "generate_mini_fixture_1000.py")], cwd=repo, check=True)
    out = tmp_path / "phenotype"
    subprocess.run([
        sys.executable, str(repo / "run_prisma.py"),
        "--manifest", "examples/mini_fixture_1000/manifest.csv",
        "--out", str(out),
        "--rank", "2",
        "--phenotype-name", "height",
        "--ld-reference-mode", "identity",
        "--allow-identity-ld",
        "--iter", "1",
        "--no_banner",
        "--quiet-blocks",
    ], cwd=repo, check=True)
    c = pd.read_csv(out / "Factor_C_Phenotypes.csv", index_col=0)
    assert list(c.index) == ["height"]
