from dyeles.utils import MoleculeStandardizer
import pickle
import math
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from pathlib import Path
import pickle
from tqdm.auto import tqdm
import gzip
from loguru import logger

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

    def score(
        self, smiles: str, return_confidence=False, use_standardizer=True
    ) -> float:
        """输入一个SMILES字符串，返回打分(float)。无效的返回None。"""
        mol = Chem.MolFromSmiles(smiles)
        if use_standardizer:
            mol = self.standardizer.standardize(mol)
        return self._score_mol(mol, return_confidence)

    def score_batch(
        self, smiles: list[str], return_confidence=False, use_standardizer=True
    ) -> list[float]:
        """输入一个SMILES列表，返回打分(float)。无效的返回None。"""
        scores = []
        for smiles in tqdm(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if use_standardizer:
                mol = self.standardizer.standardize(mol)
            score = self._score_mol(mol, return_confidence)
            if score is not None:
                scores.append(score)
        return scores
