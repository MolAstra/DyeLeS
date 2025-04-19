import pandas as pd
from pathlib import Path
from typing import List
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm
from loguru import logger
from dyeles.utils import MoleculeStandardizer, calculate_qed_properties

# 关闭 RDKit 日志
RDLogger.DisableLog("rdApp.*")


def process_molecules(
    df: pd.DataFrame, standardizer: MoleculeStandardizer, col_smiles: str = "smiles"
) -> List[dict]:
    """标准化分子并计算QED性质"""
    samples = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
        raw_smiles = row[col_smiles]
        try:
            mol = Chem.MolFromSmiles(raw_smiles)
            mol = standardizer.standardize(mol)
            if mol is None:
                logger.warning(f"Standardization failed for SMILES: {raw_smiles}")
                continue
            props = calculate_qed_properties(mol)
            sample = {
                "smiles": Chem.MolToSmiles(mol),
                "MW": props.MW,
                "ALOGP": props.ALOGP,
                "HBA": props.HBA,
                "HBD": props.HBD,
                "PSA": props.PSA,
                "ROTB": props.ROTB,
                "AROM": props.AROM,
                "ALERTS": props.ALERTS,
            }
            samples.append(sample)
        except Exception as e:
            logger.error(
                f"Error processing molecule at index {idx} with SMILES {raw_smiles}: {e}"
            )
            continue
    return samples


if __name__ == "__main__":
    data_dir = Path("./data")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 载入数据
    df_raw_path = data_dir / "moses" / "dataset_v1.csv"
    df_raw = pd.read_csv(df_raw_path)
    logger.info(f"Loaded {len(df_raw)} molecules.")

    standardizer = MoleculeStandardizer()

    # 处理分子
    sample_list = process_molecules(df_raw, standardizer, col_smiles="SMILES")

    # 保存处理后的数据
    df = (
        pd.DataFrame(sample_list)
        .drop_duplicates(subset=["smiles"])
        .reset_index(drop=True)
    )
    df.to_csv(processed_dir / "zinc.csv", index=False)
    logger.success(
        f"Saved {len(df)} processed molecules to {processed_dir / 'zinc.csv'}"
    )
