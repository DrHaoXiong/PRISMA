import pandas as pd
import numpy as np
import os
import re


def normalize_chromosome(value):
    """Normalize chromosome labels such as chr1 and 1 to integer autosomes."""
    text = str(value).strip()
    text = re.sub(r"^chr", "", text, flags=re.IGNORECASE)
    return int(text)


def _split_bed_line(line):
    line = line.strip()
    if "," in line:
        return [part.strip() for part in line.split(",")]
    return re.split(r"\s+", line)


def _looks_like_header(parts):
    if len(parts) < 3:
        return False
    lowered = [str(p).strip().lower() for p in parts[:3]]
    if lowered[0] in {"chr", "chrom", "chromosome"}:
        return True
    try:
        normalize_chromosome(parts[0])
        int(float(parts[1]))
        int(float(parts[2]))
        return False
    except Exception:
        return True


def read_ld_blocks_bed(bed_file):
    """
    Read headered or no-header BED-like LD block definitions.

    Supported delimiters are comma, tab, and whitespace. The returned DataFrame
    always contains chr, start, stop, and block_id columns.
    """
    if not os.path.exists(bed_file):
        raise FileNotFoundError(f"LD block BED file does not exist: {bed_file}")

    with open(bed_file, "r", encoding="utf-8") as handle:
        non_empty_lines = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]

    if not non_empty_lines:
        raise ValueError(f"LD block BED file is empty: {bed_file}")

    first_parts = _split_bed_line(non_empty_lines[0])
    has_header = _looks_like_header(first_parts)

    data_lines = non_empty_lines[1:] if has_header else non_empty_lines
    records = []
    for line_number, line in enumerate(data_lines, start=2 if has_header else 1):
        parts = _split_bed_line(line)
        if len(parts) < 3:
            raise ValueError(f"Invalid BED line {line_number}: expected at least 3 columns.")
        try:
            chrom = normalize_chromosome(parts[0])
            start = int(float(parts[1]))
            stop = int(float(parts[2]))
        except Exception as exc:
            raise ValueError(f"Invalid BED line {line_number}: chr/start/stop could not be parsed.") from exc
        if start >= stop:
            raise ValueError(f"Invalid BED line {line_number}: start must be < stop.")
        block_id = parts[3] if len(parts) >= 4 else f"block_{len(records)}"
        records.append((chrom, start, stop, block_id))

    if not records:
        raise ValueError(f"LD block BED file contains no block records: {bed_file}")

    return pd.DataFrame(records, columns=["chr", "start", "stop", "block_id"])

class GenomicPartitioner:
    """
    PRISMA genomic partitioner.

    Responsibilities:
    1. Read LD block definitions from a BED-like file.
    2. Receive the aligned DataFrame produced by loader.py.
    3. Yield one genomic block at a time.
    """

    def __init__(self, data_df):
        """
        Parameters:
        - data_df: aligned DataFrame containing CHR and BP columns.
        """
        self.data_df = data_df
        # Sort by genomic coordinates for efficient block queries.
        if 'CHR' in self.data_df.columns and 'BP' in self.data_df.columns:
            self.data_df = self.data_df.copy()
            self.data_df['CHR'] = self.data_df['CHR'].map(normalize_chromosome)
            self.data_df['BP'] = pd.to_numeric(self.data_df['BP'], errors='raise').astype(int)
            self.data_df = self.data_df.sort_values(['CHR', 'BP']).reset_index(drop=True)
        else:
            raise ValueError("Partitioner input must contain 'CHR' and 'BP' columns.")
        self.block_stats = {}

    def load_block_definitions(self, bed_file=None, exclude_complex_ld=True):
        """
        Load LD block definitions. If bed_file is None, generate toy blocks
        for local testing.

        Parameters:
        - exclude_complex_ld: whether to exclude complex LD regions such as
          MHC and common inversion regions.
        """
        # Complex LD regions in hg19/GRCh37 coordinates.
        COMPLEX_LD_REGIONS = [
            (6, 25000000, 35000000),   # MHC
            (8, 7000000, 13000000),    # 8p23 inversion
            (11, 45000000, 57000000),  # 11p15.5 imprinted region
        ]

        if bed_file is None:
            print("[WARNING] No BED file provided; using toy block definitions for testing.")
            blocks = pd.DataFrame({
                'chr': [1, 1, 1],
                'start': [0, 200, 400],
                'stop': [200, 400, 600],
                'block_id': ['block_1', 'block_2', 'block_3']
            })
            return blocks
        else:
            blocks = read_ld_blocks_bed(bed_file)
            n_loaded = len(blocks)

            if exclude_complex_ld:
                n_before = len(blocks)
                for chrom, reg_start, reg_stop in COMPLEX_LD_REGIONS:
                    mask = ~((blocks['chr'] == chrom) &
                            (blocks['stop'] > reg_start) &
                            (blocks['start'] < reg_stop))
                    blocks = blocks[mask]
                n_after = len(blocks)
                print(f"[INFO] Excluded {n_before - n_after} complex LD blocks.")
                n_removed_complex = n_before - n_after
            else:
                n_removed_complex = 0

            blocks = blocks.reset_index(drop=True)
            self.block_stats = {
                "n_blocks_loaded": int(n_loaded),
                "n_blocks_removed_complex_ld": int(n_removed_complex),
                "n_blocks_used": int(len(blocks)),
            }
            print(
                "[INFO] LD blocks loaded: "
                f"{self.block_stats['n_blocks_loaded']}; "
                f"removed complex LD: {self.block_stats['n_blocks_removed_complex_ld']}; "
                f"used: {self.block_stats['n_blocks_used']}"
            )
            return blocks

    def iter_blocks(self, blocks_df):
        """
        Yield one non-empty block-specific DataFrame at a time.
        """
        total_blocks = len(blocks_df)
        
        for idx, row in blocks_df.iterrows():
            c = normalize_chromosome(row['chr'])
            start = int(row['start'])
            stop = int(row['stop'])
            bid = row.get('block_id', f"block_{idx}")

            # Left-closed, right-open interval: [start, stop).
            mask = (self.data_df['CHR'] == c) & \
                   (self.data_df['BP'] >= start) & \
                   (self.data_df['BP'] < stop)
            
            sub_df = self.data_df[mask].copy()

            # Yield only non-empty blocks.
            if len(sub_df) > 0:
                if os.environ.get('PRISMA_QUIET_BLOCKS') != '1':
                    print(f"[INFO] Block {bid} (Chr{c}: {start}-{stop}) contains {len(sub_df)} SNPs.")
                yield bid, sub_df
            else:
                # Empty block.
                pass

# ================= Test block =================
if __name__ == "__main__":
    # 1. Create toy aligned data.
    # rs1(100), rs2(200), rs3(300), rs4(400)
    # Toy blocks are [0-200), [200-400), [400-600).
    
    df_aligned = pd.DataFrame({
        'SNP': ['rs1', 'rs2', 'rs3', 'rs4'],
        'CHR': [1, 1, 1, 1],
        'BP':  [100, 200, 300, 400], # Boundary-condition example.
        'Z_gwas': [5.0, 3.0, 2.0, 4.0],
        'Z_eqtl': [5.0, -5.0, 2.5, -4.0]
    })

    print("Toy full dataset:\n", df_aligned)
    print("\n" + "="*30 + "\n")

    # 2. Initialize partitioner.
    partitioner = GenomicPartitioner(df_aligned)
    
    # 3. Load toy block definitions.
    blocks_def = partitioner.load_block_definitions()
    print("Block definitions:\n", blocks_def)
    print("\n" + "="*30 + "\n")

    # 4. Stream blocks.
    for block_id, block_data in partitioner.iter_blocks(blocks_def):
        print(f"Main received: {block_id}")
        print(block_data)
        print("-" * 20)
        
    # Expected:
    # block_1 (0-200): rs1 (BP 100)
    # block_2 (200-400): rs2 (BP 200), rs3 (BP 300)
    # block_3 (400-600): rs4 (BP 400)
