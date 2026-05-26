import pandas as pd

from partition import read_ld_blocks_bed


def test_headered_bed_loads_all_rows(tmp_path):
    bed = tmp_path / "blocks_header.bed"
    bed.write_text("chr,start,stop\nchr1,100,200\nchr1,200,300\n", encoding="utf-8")
    blocks = read_ld_blocks_bed(bed)
    assert len(blocks) == 2
    assert list(blocks.columns) == ["chr", "start", "stop", "block_id"]
    assert blocks.iloc[0]["start"] == 100


def test_comma_no_header_bed_loads_all_rows(tmp_path):
    bed = tmp_path / "blocks_comma_no_header.bed"
    bed.write_text("chr1,100,200\nchr1,200,300\n", encoding="utf-8")
    blocks = read_ld_blocks_bed(bed)
    assert len(blocks) == 2
    assert blocks.iloc[0]["chr"] == 1
    assert blocks.iloc[1]["stop"] == 300
