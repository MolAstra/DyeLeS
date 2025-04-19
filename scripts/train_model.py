from absl import app, flags
from absl.flags import FLAGS
from rdkit import Chem
from loguru import logger
from rdkit.Chem import rdFingerprintGenerator
import math
import gzip
import pickle
from typing import Dict, Set, Any, Iterator, List, Tuple
import sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Define command line flags
flags.DEFINE_string(
    "dye_csv", None, "CSV file containing natural product SMILES", short_name="d"
)
flags.DEFINE_string(
    "non_dye_csv",
    None,
    "CSV file containing non-natural product SMILES",
    short_name="n",
)
flags.DEFINE_string("smiles_column", "smiles", "Column name containing SMILES strings")
flags.DEFINE_string("out_model_file", None, "Output model filename", short_name="o")
flags.DEFINE_integer(
    "radius", 2, "Morgan fingerprint radius for RDKit", lower_bound=1, upper_bound=5
)
flags.DEFINE_integer(
    "fp_size", 2048, "Fingerprint length", lower_bound=128, upper_bound=4096
)
flags.DEFINE_float(
    "alpha", 0.001, "Smoothing factor (Laplace smoothing)", lower_bound=0.0
)
flags.DEFINE_integer(
    "chunk_size", 50000, "Number of molecules to process at once", lower_bound=1000
)
flags.DEFINE_integer(
    "report_interval", 100000, "Progress report interval", lower_bound=1000
)

# Mark required flags
flags.mark_flag_as_required("dye_csv")
flags.mark_flag_as_required("non_dye_csv")
flags.mark_flag_as_required("out_model_file")


def read_smiles_chunks(
    csv_path: str, smiles_column: str, chunk_size: int
) -> Iterator[List[str]]:
    """Generator that yields chunks of SMILES strings from a CSV file.

    Args:
        csv_path: Path to CSV file
        smiles_column: Name of column containing SMILES
        chunk_size: Number of rows to read at once

    Yields:
        Lists of SMILES strings in chunks
    """
    logger.info(f"Reading SMILES from {csv_path}")
    try:
        for chunk in pd.read_csv(
            csv_path, chunksize=chunk_size, usecols=[smiles_column]
        ):
            if smiles_column not in chunk.columns:
                raise ValueError(f"Column '{smiles_column}' not found in CSV file")
            yield chunk[smiles_column].dropna().tolist()
    except Exception as e:
        logger.error(f"Failed to read CSV file: {str(e)}")
        raise


def process_smiles_chunk(
    smiles_list: List[str], fp_gen: rdFingerprintGenerator.FingerprintGenerator64
) -> Dict[int, int]:
    """Process a chunk of SMILES strings and count fingerprint bits.

    Args:
        smiles_list: List of SMILES strings to process
        fp_gen: Fingerprint generator instance

    Returns:
        Tuple of (bit_count_dict, valid_count) where:
        - bit_count_dict: {bit_index: occurrence_count}
        - valid_count: Number of successfully processed molecules
    """
    bit_count = defaultdict(int)
    valid_count = 0

    for smiles in tqdm(smiles_list, desc="Processing SMILES with chunk"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        valid_count += 1
        fp = fp_gen.GetSparseCountFingerprint(mol)
        for bit, count in fp.GetNonzeroElements().items():
            bit_count[bit] += count

    return bit_count, valid_count


def count_fingerprints(
    csv_path: str,
    smiles_column: str,
    fp_gen: rdFingerprintGenerator.FingerprintGenerator64,
    chunk_size: int,
    report_interval: int,
) -> Dict[int, int]:
    """Count fingerprint bits from SMILES in CSV file using chunked processing.

    Args:
        csv_path: Path to CSV file
        smiles_column: Name of column containing SMILES
        fp_gen: Fingerprint generator
        chunk_size: Number of molecules to process at once
        report_interval: How often to report progress

    Returns:
        Tuple of (bit_count_dict, total_molecules) where:
        - bit_count_dict: {bit_index: occurrence_count}
        - total_molecules: Total valid molecules processed
    """
    all_bit_counts = defaultdict(int)
    total_molecules = 0
    total_processed = 0

    for smiles_chunk in read_smiles_chunks(csv_path, smiles_column, chunk_size):
        bit_count, valid_count = process_smiles_chunk(smiles_chunk, fp_gen)

        # Merge counts
        for bit, count in bit_count.items():
            all_bit_counts[bit] += count

        total_molecules += valid_count
        total_processed += len(smiles_chunk)

        if total_processed % report_interval < chunk_size or total_processed == len(
            smiles_chunk
        ):
            logger.info(
                f"Processed {total_processed:,} rows ({total_molecules:,} valid molecules)"
            )

    logger.success(f"Completed processing {total_processed:,} total rows")
    logger.success(f"Found {total_molecules:,} valid molecules")
    logger.success(f"Collected {len(all_bit_counts):,} unique fingerprint bits")

    return dict(all_bit_counts), total_molecules


def calculate_fscore(
    dye_count: Dict[int, int], non_dye_count: Dict[int, int], alpha: float
) -> Dict[int, float]:
    """Calculate log-likelihood ratio scores for fingerprint bits."""
    sum_dye = sum(dye_count.values())
    sum_non_dye = sum(non_dye_count.values())

    all_bits = set(dye_count.keys()).union(set(non_dye_count.keys()))
    total_bits = len(all_bits)

    fscore = {}
    for bit_id in all_bits:
        np_x = dye_count.get(bit_id, 0)
        non_np_x = non_dye_count.get(bit_id, 0)

        p_np = (np_x + alpha) / (sum_dye + alpha * total_bits)
        p_non_np = (non_np_x + alpha) / (sum_non_dye + alpha * total_bits)

        fscore[bit_id] = math.log10(p_np / p_non_np)

    return fscore


def main(args: Any) -> None:
    """Main training function for the fingerprint scoring model."""
    try:
        # Initialize fingerprint generator
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=FLAGS.radius, fpSize=FLAGS.fp_size
        )

        # Process natural product data
        logger.info("Processing natural product dataset...")
        dye_count, dye_mol_count = count_fingerprints(
            FLAGS.dye_csv,
            FLAGS.smiles_column,
            fp_gen,
            FLAGS.chunk_size,
            FLAGS.report_interval,
        )

        # Process non-natural product data
        logger.info("Processing non-natural product dataset...")
        non_dye_count, non_dye_mol_count = count_fingerprints(
            FLAGS.non_dye_csv,
            FLAGS.smiles_column,
            fp_gen,
            FLAGS.chunk_size,
            FLAGS.report_interval,
        )

        # Calculate fingerprint scores
        logger.info("Calculating fingerprint scores...")
        fscore = calculate_fscore(dye_count, non_dye_count, FLAGS.alpha)

        # Save model with metadata
        logger.info(f"Saving model to: {FLAGS.out_model_file}")
        with gzip.open(FLAGS.out_model_file, "wb") as f:
            pickle.dump(
                {
                    "fscore": fscore,
                    "metadata": {
                        "dye_mol_count": dye_mol_count,
                        "non_dye_mol_count": non_dye_mol_count,
                        "radius": FLAGS.radius,
                        "fp_size": FLAGS.fp_size,
                        "alpha": FLAGS.alpha,
                        "model_type": "fingerprint_log_likelihood",
                        "smiles_column": FLAGS.smiles_column,
                        "total_bits": len(fscore),
                    },
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        logger.success("Training complete! Model saved successfully.")

    except Exception as e:
        logger.error(f"Runtime error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    app.run(main)
