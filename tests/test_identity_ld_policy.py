import json
import subprocess
import sys
from pathlib import Path


def _generate_fixture(repo: Path) -> None:
    subprocess.run(
        [sys.executable, str(repo / "examples" / "generate_mini_fixture_1000.py")],
        cwd=repo,
        check=True,
    )


def _write_real_style_manifest(repo: Path, manifest_path: Path) -> None:
    fixture = repo / "examples" / "mini_fixture_1000"
    manifest_path.write_text(
        "\n".join(
            [
                "type,name,path",
                f"gwas,synthetic_trait,{fixture / 'gwas.tsv'}",
                f"eqtl,retina,{fixture / 'eqtl_retina.tsv'}",
                f"eqtl,blood,{fixture / 'eqtl_blood.tsv'}",
                f"eqtl,artery,{fixture / 'eqtl_artery.tsv'}",
                f"bed,ld_blocks,{fixture / 'ld_blocks_header.bed'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_auto_without_bfile_fails_for_real_style_manifest(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    _generate_fixture(repo)
    manifest = tmp_path / "manifest.csv"
    _write_real_style_manifest(repo, manifest)

    result = subprocess.run(
        [
            sys.executable,
            str(repo / "run_prisma.py"),
            "--manifest",
            str(manifest),
            "--out",
            str(tmp_path / "should_fail_auto_no_bfile"),
            "--rank",
            "2",
            "--ld-reference-mode",
            "auto",
            "--iter",
            "1",
            "--no_banner",
            "--quiet-blocks",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "Identity Laplacian" in (result.stdout + result.stderr)
    assert "--bfile" in (result.stdout + result.stderr)


def test_auto_without_bfile_allowed_for_examples_and_recorded(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    _generate_fixture(repo)
    out = tmp_path / "auto_examples_identity"

    subprocess.run(
        [
            sys.executable,
            str(repo / "run_prisma.py"),
            "--manifest",
            "examples/mini_fixture_1000/manifest.csv",
            "--out",
            str(out),
            "--rank",
            "2",
            "--ld-reference-mode",
            "auto",
            "--iter",
            "1",
            "--no_banner",
            "--quiet-blocks",
        ],
        cwd=repo,
        check=True,
    )

    report = json.loads((out / "qc_report.json").read_text(encoding="utf-8"))
    assert report["run_configuration"]["identity_ld_active"] is True
    assert report["run_configuration"]["identity_ld_reason"] == "examples_or_tests_manifest"
    assert any("Identity Laplacian" in warning for warning in report["warnings"])


def test_plink_mode_without_bfile_fails_clearly(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    _generate_fixture(repo)

    result = subprocess.run(
        [
            sys.executable,
            str(repo / "run_prisma.py"),
            "--manifest",
            "examples/mini_fixture_1000/manifest.csv",
            "--out",
            str(tmp_path / "should_fail_plink_no_bfile"),
            "--rank",
            "2",
            "--ld-reference-mode",
            "plink",
            "--iter",
            "1",
            "--no_banner",
            "--quiet-blocks",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "--ld-reference-mode plink requires --bfile" in (result.stdout + result.stderr)
