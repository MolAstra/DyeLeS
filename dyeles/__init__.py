from typing import Iterable
from dyeles.utils import MoleculeStandardizer
import pickle
import math
from rdkit import Chem, RDLogger
import rdkit
from rdkit.Chem import rdMolDescriptors
from pathlib import Path
import pickle
from tqdm.auto import tqdm
import gzip
from loguru import logger
from typing import Tuple, List, Union

# 关闭 RDKit 日志
RDLogger.DisableLog("rdApp.*")


class DyeLeS:
    def __init__(self, model_file=None):
        """初始化，加载模型。默认使用当前脚本目录下的 publicnp.model.gz"""
        if model_file is None:
            model_file = Path(__file__).parent / "resources" / "dye.model.gz"

        self.fscore = pickle.load(gzip.open(model_file, "rb"))["fscore"]
        self.standardizer = MoleculeStandardizer()

    def _score_mol(self, mol, return_confidence=False):
        """内部方法，给定一个Mol对象，计算分数"""
        if mol is None:
            return None

        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        bits = fp.GetNonzeroElements()

        score = 0.0
        bits_found = 0
        for bit in bits:
            if bit in self.fscore:
                bits_found += 1
                score += self.fscore[bit]

        try:
            score /= float(mol.GetNumAtoms())
            confidence = float(bits_found / len(bits))
        except ZeroDivisionError:
            return None

        if score > 4:
            score = 4.0 + math.log10(score - 4.0 + 1.0)
        elif score < -4:
            score = -4.0 - math.log10(-4.0 - score + 1.0)

        if return_confidence:
            return score, confidence
        else:
            return score

    def __call__(
        self,
        inputs: Union[str, Chem.Mol, Iterable[str], Iterable[Chem.Mol]],
        return_confidence: bool = False,
        use_standardizer: bool = True,
        progress: bool = True,
    ) -> Union[float, Tuple[float, float], List[float], List[Tuple[float, float]]]:
        """
        支持直接调用对象，对单个或多个SMILES或Mol对象进行打分

        Args:
            input: 输入可以是以下类型之一：
                - SMILES 字符串
                - RDKit Mol 对象
                - SMILES 字符串列表
                - RDKit Mol 对象列表
            return_confidence: 是否返回置信度分数
            use_standardizer: 是否使用标准化器处理分子

        Returns:
            单个分子时返回分数（或分数+置信度元组）
            多个分子时返回分数列表（或元组列表）
        """
        # 处理单个输入的情况
        if isinstance(inputs, (str, Chem.Mol)):
            if isinstance(inputs, str):
                mol = Chem.MolFromSmiles(inputs)
                if mol is None:
                    logger.error(f"Invalid SMILES: {inputs}")
                    return 0.0
            else:  # 已经是 Mol 对象
                mol = inputs

            if use_standardizer:
                mol = self.standardizer.standardize(mol)
            return self._score_mol(mol, return_confidence)

        # 处理多个输入的情况
        elif isinstance(inputs, Iterable):
            results = []
            for item in tqdm(inputs, desc="Scoring molecules", disable=not progress):
                if isinstance(item, str):
                    mol = Chem.MolFromSmiles(item)
                    if mol is None:
                        logger.error(f"Invalid SMILES: {item}")
                        results.append(0.0)
                        continue
                else:  # 已经是 Mol 对象
                    mol = item

                if use_standardizer:
                    mol = self.standardizer.standardize(mol)
                score = self._score_mol(mol, return_confidence)
                results.append(score)
            return results

        else:
            raise TypeError(
                "Input must be SMILES string, Mol object, or their iterable"
            )
