from partition import read_ld_blocks_bed


def test_whitespace_bed_loads(tmp_path):
    bed = tmp_path / "blocks_space.bed"
    bed.write_text("chr start stop\n1 100 200\n1 200 300\n", encoding="utf-8")
    blocks = read_ld_blocks_bed(bed)
    assert len(blocks) == 2
    assert blocks["chr"].tolist() == [1, 1]
