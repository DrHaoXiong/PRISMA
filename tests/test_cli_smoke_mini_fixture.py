import json
import subprocess
import sys
from pathlib import Path


def test_cli_smoke_mini_fixture_outputs(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(repo / "examples" / "generate_mini_fixture_1000.py")], cwd=repo, check=True)
    out = tmp_path / "mini_fixture_test"
    subprocess.run([
        sys.executable, str(repo / "run_prisma.py"),
        "--manifest", "examples/mini_fixture_1000/manifest.csv",
        "--out", str(out),
        "--rank", "auto",
        "--max-rank", "3",
        "--phenotype-name", "synthetic_trait",
        "--ld-reference-mode", "identity",
        "--allow-identity-ld",
        "--iter", "2",
        "--no_banner",
        "--quiet-blocks",
    ], cwd=repo, check=True)
    for name in [
        "Factor_A_SNPs.csv",
        "Factor_B_Tissues.csv",
        "Factor_C_Phenotypes.csv",
        "qc_report.json",
        "qc_summary.csv",
        "qc_report.tsv",
        "qc_report.txt",
        "rank_diagnostics.csv",
        "rank_selection.json",
    ]:
        assert (out / name).exists()
    selection = json.loads((out / "rank_selection.json").read_text(encoding="utf-8"))
    assert "selected_rank" in selection
