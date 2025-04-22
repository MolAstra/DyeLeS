import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import joblib
import pandas as pd
from loguru import logger
from pathlib import Path

class SimplePredictor:
    def __init__(self, model_path: str = None):
        """初始化 Predictor 类并加载训练好的模型"""
        if model_path is None:
            model_path = Path(__file__).parent / "resources" / "lightgbm.pkl"
        else:
            model_path = Path(model_path)
        self.model = self.load_model(model_path)
        self.n_bits = 2048  # 默认摩根指纹长度

    def smiles_to_morgan_fingerprint(self, smiles: str, radius: int = 2, n_bits: int = 2048):
        """将 SMILES 字符串转化为摩根指纹"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)  # 无效 SMILES 返回零向量
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fingerprint)

    def load_model(self, model_path: str):
        """加载训练好的模型"""
        logger.info(f"Loading the trained model from {model_path}...")
        with open(model_path, "rb") as f:
            model = joblib.load(f)
        return model

    def predict(self, smiles_list: list):
        """预测多个 SMILES 字符串对应的 absorption, emission, stokes_shift 和 quantum_yield"""
        logger.debug(f"Predicting for {len(smiles_list)} SMILES.")
        
        # 1. 将 SMILES 字符串转化为摩根指纹
        fingerprints = [self.smiles_to_morgan_fingerprint(smiles) for smiles in smiles_list]

        # 2. 转换为 DataFrame，并添加列名（避免特征名警告）
        fingerprints_df = pd.DataFrame(fingerprints, columns=[f"bit_{i}" for i in range(self.n_bits)])

        # 3. 批量预测
        predictions = self.model.predict(fingerprints_df)

        return predictions


if __name__ == "__main__":
    # 加载已训练并保存的模型
    model_path = "../dyeles/resources/lightgbm.pkl"  # 请替换为模型文件的实际路径
    predictor = SimplePredictor(model_path)

    # 输入多个 SMILES 进行批量预测
    smiles_list = ["CCO", "CC(=O)O", "c1ccc(C(=O)O)cc1"]  # 请替换为实际的 SMILES 字符串列表
    predictions = predictor.predict(smiles_list)

    # 输出预测结果
    for i, (absorption, emission, stokes_shift, quantum_yield) in enumerate(predictions):
        print(f"SMILES: {smiles_list[i]}")
        print(f"Predicted Absorption: {absorption}")
        print(f"Predicted Emission: {emission}")
        print(f"Predicted Stokes Shift: {stokes_shift}")
        print(f"Predicted Quantum Yield: {quantum_yield}")
        print()
