from dataclasses import dataclass
import math
import time
import tempfile
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from pathlib import Path
from typing import Literal, Optional
from dyeles import DyeLeS
from dyeles.ml import SimplePredictor
from loguru import logger
import yaml

CONFIG_DIR = Path(__file__).parent / "configs"


@dataclass
class PredictResults:
    smiles: str
    molecular_weight: float
    dye_likeness_score: float
    stokes_shift: float
    quantum_yield: float
    absorption: float
    emission: float

    def to_json(self):
        return {
            "smiles": self.smiles,
            "molecular_weight": self.molecular_weight,
            "dye_likeness_score": self.dye_likeness_score,
            "stokes_shift": self.stokes_shift,
            "quantum_yield": self.quantum_yield,
            "absorption": self.absorption,
            "emission": self.emission,
        }


class Predictor:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.db = self.load_db()
        self.predictor = SimplePredictor()
        self.dyeles = DyeLeS()

    def load_db(self):
        df = pd.read_csv(self.data_path)
        df.rename(
            columns={
                "Chromophore": "smiles",
                "Tag": "tag",
                "Solvent": "solvent",
                "Absorption max (nm)": "absorption",
                "Emission max (nm)": "emission",
                "Lifetime (ns)": "lifetime",
                "Quantum yield": "quantum_yield",
                "log(e/mol-1 dm3 cm-1)": "log_e",
                "abs FWHM (cm-1)": "abs_fwhm",
                "emi FWHM (cm-1)": "emi_fwhm",
                "abs FWHM (nm)": "abs_fwhm_nm",
                "emi FWHM (nm)": "emi_fwhm_nm",
                "Molecular weight (g mol-1)": "molecular_weight",
                "Reference": "reference",
            },
            inplace=True,
        )
        return df.copy()

    def get_canonical_smiles(self, smiles: str) -> Optional[str]:
        """获取标准SMILES"""
        try:
            return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        except Exception as e:
            logger.error(f"Error getting canonical SMILES for {smiles}: {e}")
            return None

    def search_db(self, smiles: str) -> dict:
        """从数据库中查找是否已有记录"""
        try:
            canonical_smiles = self.get_canonical_smiles(smiles)
            if canonical_smiles is None:
                return None
        
            hit = self.db[self.db["smiles"] == canonical_smiles]

            if hit.empty:
                return None
            if len(hit) > 1:
                logger.warning(f"Multiple hits found for {smiles}, using first one")
                hit = hit.iloc[0]
            logger.info(hit)
            return {
                "smiles": hit["smiles"],
                "molecular_weight": hit.get("molecular_weight"),
                "dye_likeness_score": hit.get("dye_likeness_score"),
                "stokes_shift": hit.get("stokes_shift"),
                "quantum_yield": hit.get("quantum_yield"),
                "absorption": hit.get("absorption"),
                "emission": hit.get("emission"),
            }
        except Exception as e:
            logger.error(f"Error searching DB for {smiles}: {e}")
            return None

    def predict_dyeles(self, smiles: str) -> dict:
        try:
            return {"dye_likeness_score": self.dyeles(smiles)}
        except Exception as e:
            logger.error(f"Error predicting dye likeness score for {smiles}: {e}")
            return None

    def predict_properties(self, smiles: str) -> dict:
        properties = self.predictor.predict([smiles])[0]
        try:
            return {
                "absorption": properties[0],
                "emission": properties[1],
                "stokes_shift": properties[2],
                "quantum_yield": properties[3],
            }
        except Exception as e:
            logger.error(f"Error predicting properties for {smiles}: {e}")
            return None

    def calculate_molecular_weight(self, smiles: str) -> float:
        # 使用 RDKit 从 SMILES 字符串生成分子
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            logger.error(f"Invalid SMILES string: {smiles}")
            return None

        # 计算分子量
        molecular_weight = Chem.rdMolDescriptors.CalcExactMolWt(mol)
        return molecular_weight

    def predict(self, smiles: str) -> dict:
        results = {
            "smiles": smiles,
            "dye_likeness_score": None,
            "molecular_weight": None,
            "absorption": None,
            "emission": None,
            "stokes_shift": None,
            "quantum_yield": None,
        }
        dye_likeness_score = self.predict_dyeles(smiles)
        if dye_likeness_score is not None:
            logger.info(f"Dye likeness score: {dye_likeness_score}")
            results.update(dye_likeness_score)
        properties = self.predict_properties(smiles)
        if properties is not None:
            logger.info(f"Properties: {properties}")
            results.update(properties)
        molecular_weight = self.calculate_molecular_weight(smiles)
        if molecular_weight is not None:
            results.update({"molecular_weight": molecular_weight})

        # 更新数据库中的结果
        results_db = self.search_db(smiles)
        if results_db is not None:
            for k in results.keys():
                if self.is_valid(results_db.get(k)):
                    logger.info(f"Updating {k} from {results_db[k]} to {results[k]}")
                    results[k] = results_db[k]

        # fix stokes shift
        results["stokes_shift"] = results["emission"] - results["absorption"]
        return results

    def is_valid(self, value):
        return value is not None and (
            isinstance(value, (int, float)) and math.isfinite(value)
        )


def predict_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        canonical_smiles = Chem.MolToSmiles(mol)
        if mol:
            img = Draw.MolToImage(mol)
        else:
            img = None

        response = requests.post(
            "http://localhost:8000/predict", json={"smiles": smiles}, timeout=10
        )
        response.raise_for_status()

        response_data = response.json()
        status = response_data["status"]
        print(type(response_data))
        print(response_data)
        if status == "success":
            result_df = pd.DataFrame([response_data])
            result_df["Canonical SMILES"] = canonical_smiles

            return f"Done ✅ - Canonical SMILES: {canonical_smiles}", result_df, img
    except Exception as e:
        return f"Failed ❌ - {e}", None, None


def load_config(type: Literal["server", "web"]):
    config_file = CONFIG_DIR / f"{type}.yaml"
    logger.info(f"Loading config from {config_file}")
    try:
        with config_file.open("r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Config file {config_file} not found")


def clean_json(obj):
    """清除 dict/json 中不合法的浮点数 (NaN, inf)"""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(i) for i in obj]
    elif isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None  # 转为 None
    return obj


def download_results(results_df):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", newline="", suffix=".csv"
    ) as temp_file:
        results_df.to_csv(temp_file, index=False)
        return temp_file.name
