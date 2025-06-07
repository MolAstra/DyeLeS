from typing import Union, Tuple, NamedTuple
from rdkit.Chem import QED
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class MoleculeStandardizer:
    def __init__(self):
        self.normalizer = rdMolStandardize.Normalizer()
        self.uncharger = rdMolStandardize.Uncharger()
        self.fragment_chooser = rdMolStandardize.LargestFragmentChooser()

    def standardize(self, mol):
        """用标准rdkit方法做分子标准化，包括电荷归一化和移除小离子"""
        if mol is None:
            return None

        # 1. 规范化结构（比如修正奇怪的分子形式，标准化）
        mol = self.normalizer.normalize(mol)

        # 2. 电荷归一化（把正负离子中和）
        mol = self.uncharger.uncharge(mol)

        # 3. 保留最大片段（比如只保留药物分子，去掉小盐）
        mol = self.fragment_chooser.choose(mol)
        
        # 4. Remove chirality
        # mol = Chem.RemoveStereochemistry(mol)

        # 5. 再次清理，确保规范
        Chem.SanitizeMol(mol)

        return mol


def calculate_qed_properties(
    mol, return_qed=False
) -> Union[NamedTuple, Tuple[NamedTuple, float]]:
    """return a namedtuple of properties
    MW,ALOGP,HBA,HBD,PSA,ROTB,AROM,ALERTS
    """
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    props = QED.properties(mol)
    qed = QED.qed(mol)
    if return_qed:
        return props, qed
    else:
        return props

