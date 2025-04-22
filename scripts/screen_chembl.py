from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dyeles.ml import SimplePredictor
from dyeles import DyeLeS
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")

# 参数
data_dir = Path("./data")
chembl_data_path = data_dir / "chembl" / "chembl_35_chemreps.txt"
predictor = SimplePredictor()
scorer = DyeLeS()

batch_size = 512  # 设置每批次处理的数量
all_valid = []

# 一次遍历整个文件
df = pd.read_csv(chembl_data_path, sep="\t")

df = df.dropna(subset=["canonical_smiles"])
df = df[df["canonical_smiles"].str.strip() != ""]
df = df[["chembl_id", "canonical_smiles"]]

num_batches = (len(df) + batch_size - 1) // batch_size

for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
    batch_df = df.iloc[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    smiles_list = batch_df["canonical_smiles"].tolist()

    try:
        predictions = predictor.predict(smiles_list)
    except Exception as e:
        logger.error(f"Prediction error in batch {batch_idx}: {e}")
        continue

    try:
        scores = scorer(smiles_list, progress=False)
    except Exception as e:
        logger.error(f"Scoring error in batch {batch_idx}: {e}")
        continue

    # 转成 DataFrame
    predictions_df = pd.DataFrame(
        predictions,
        columns=["absorption", "emission", "stokes_shift", "quantum_yield"],
    )
    scores_df = pd.DataFrame(scores, columns=["score"])

    assert len(batch_df) == len(predictions_df) == len(scores_df), \
        f"Batch size mismatch: {len(batch_df)} != {len(predictions_df)} != {len(scores_df)}"

    combined = pd.concat(
        [batch_df.reset_index(drop=True), predictions_df, scores_df], axis=1
    )

    # 只保留符合要求的分子
    filtered = combined[
        (combined["stokes_shift"] > 76)
        & (combined["quantum_yield"] > 0.34)
        & (combined["score"] > 0.5)
    ]

    if not filtered.empty:
        all_valid.append(filtered)

# 合并所有筛选到的结果
if all_valid:
    final_full_result = pd.concat(all_valid, ignore_index=True)
    # 保存完整符合条件的数据
    final_full_result.to_csv("data/chembl/chembl_dyes.csv", index=False)
    logger.info(f"保存了 {len(final_full_result)} 条符合条件的数据到 chembl_dyes.csv")
else:
    logger.info("没有符合条件的数据。")
