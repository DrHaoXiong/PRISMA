import pandas as pd
import numpy as np
import os

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
            self.data_df = self.data_df.sort_values(['CHR', 'BP']).reset_index(drop=True)
        else:
            raise ValueError("Partitioner input must contain 'CHR' and 'BP' columns.")

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
            blocks = pd.read_csv(bed_file, sep='\t', names=['chr', 'start', 'stop'], skiprows=1)
            # Clean chr column: remove optional "chr" prefix and convert to integer.
            blocks['chr'] = blocks['chr'].astype(str).str.replace('chr', '').str.strip().astype(int)
            blocks['start'] = blocks['start'].astype(int)
            blocks['stop'] = blocks['stop'].astype(int)

            if exclude_complex_ld:
                n_before = len(blocks)
                for chrom, reg_start, reg_stop in COMPLEX_LD_REGIONS:
                    mask = ~((blocks['chr'] == chrom) &
                            (blocks['stop'] > reg_start) &
                            (blocks['start'] < reg_stop))
                    blocks = blocks[mask]
                n_after = len(blocks)
                print(f"[INFO] Excluded {n_before - n_after} complex LD blocks.")

            return blocks

    def iter_blocks(self, blocks_df):
        """
        Yield one non-empty block-specific DataFrame at a time.
        """
        total_blocks = len(blocks_df)
        
        for idx, row in blocks_df.iterrows():
            c = row['chr']
            start = row['start']
            stop = row['stop']
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
