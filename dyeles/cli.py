from absl import app, flags
import pandas as pd
from dyeles import DyeLeS
from loguru import logger
from rdkit import Chem, RDLogger
import os

# 关闭 RDKit 日志
RDLogger.DisableLog("rdApp.*")

# 定义命令行参数
FLAGS = flags.FLAGS

# 主参数定义 + 短命令
flags.DEFINE_string("input", None, "Path to input CSV file containing SMILES. (Short: -i)")
flags.DEFINE_string("i", None, "Alias for --input")

flags.DEFINE_string("output", None, "Path to output CSV file to save results. (Short: -o)")
flags.DEFINE_string("o", None, "Alias for --output")

flags.DEFINE_boolean("use_standardizer", True, "Whether to use RDKit standardizer before scoring. Default: True.")
flags.DEFINE_boolean("return_confidence", False, "Whether to output confidence scores. Default: False.")

flags.DEFINE_boolean("force_write", False, "Force overwrite output file if it exists. (Short: -f)")
flags.DEFINE_boolean("f", False, "Alias for --force_write")

# 必须参数
flags.mark_flag_as_required("input")
flags.mark_flag_as_required("output")


def _main(argv):
    del argv  # unused

    # 处理短参数别名
    input_path = FLAGS.input or FLAGS.i
    output_path = FLAGS.output or FLAGS.o
    force_write = FLAGS.force_write or FLAGS.f

    if input_path is None or output_path is None:
        logger.error("Error: Both --input (-i) and --output (-o) are required.")
        return

    # 如果输出文件存在且没有force写入
    if os.path.exists(output_path) and not force_write:
        logger.error(f"Output file {output_path} already exists. Use --force_write (-f) to overwrite.")
        return

    # 读取输入CSV
    df = pd.read_csv(input_path)

    if "smiles" not in df.columns:
        logger.error("Error: Input CSV must contain a 'smiles' column.")
        return

    logger.info(f"Loaded {len(df)} molecules from {input_path}")

    scorer = DyeLeS()

    if not FLAGS.return_confidence:
        scores = scorer.score_batch(
            df["smiles"].tolist(), use_standardizer=FLAGS.use_standardizer
        )
        df["score"] = scores
    else:
        scores = scorer.score_batch(
            df["smiles"].tolist(),
            use_standardizer=FLAGS.use_standardizer,
            return_confidence=True,
        )
        df["score"] = [score[0] for score in scores]
        df["confidence"] = [score[1] for score in scores]

    df.to_csv(output_path, index=False)
    logger.success(f"Scoring complete. Results saved to {output_path}")


def main():
    app.run(_main)