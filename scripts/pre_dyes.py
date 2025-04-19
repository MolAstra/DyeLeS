import pandas as pd
from pathlib import Path
from typing import List
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm
from loguru import logger
from dyeles.utils import MoleculeStandardizer, calculate_qed_properties

# 关闭 RDKit 日志
RDLogger.DisableLog("rdApp.*")


def load_and_merge_dye_data(data_dir: Path) -> pd.DataFrame:
    """加载和合并荧光染料数据集"""
    fluor_data_dir = data_dir / "dyes"
    files = [
        "Dataset_Consolidation_canonicalized.csv",
        "Dataset_Cyanine_canonicalized.csv",
        "Dataset_Xanthene_canonicalized.csv",
    ]
    dfs = [pd.read_csv(fluor_data_dir / f) for f in files]
    merged_df = pd.concat(dfs).drop_duplicates(subset=["smiles"])
    logger.info(f"Loaded {len(merged_df)} unique SMILES.")
    return merged_df


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
    df_raw = load_and_merge_dye_data(data_dir)
    standardizer = MoleculeStandardizer()

    # 处理分子
    sample_list = process_molecules(df_raw, standardizer, col_smiles="smiles")

    # 保存处理后的数据
    df = (
        pd.DataFrame(sample_list)
        .drop_duplicates(subset=["smiles"])
        .reset_index(drop=True)
    )
    df.to_csv(processed_dir / "dyes.csv", index=False)
    logger.success(
        f"Saved {len(df)} processed molecules to {processed_dir / 'dyes.csv'}"
    )
