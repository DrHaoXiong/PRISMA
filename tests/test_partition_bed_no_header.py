import pytest

from partition import read_ld_blocks_bed


def test_no_header_bed_does_not_drop_first_row(tmp_path):
    bed = tmp_path / "blocks_no_header.bed"
    bed.write_text("chr1\t100\t200\nchr1\t200\t300\n", encoding="utf-8")
    blocks = read_ld_blocks_bed(bed)
    assert len(blocks) == 2
    assert blocks.iloc[0]["start"] == 100


def test_invalid_bed_raises_value_error(tmp_path):
    bed = tmp_path / "bad.bed"
    bed.write_text("chr1\t300\t200\n", encoding="utf-8")
    with pytest.raises(ValueError, match="start must be < stop"):
        read_ld_blocks_bed(bed)
