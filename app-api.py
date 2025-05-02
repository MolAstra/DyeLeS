from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from dyeles import DyeLeS
from pathlib import Path
import click
import yaml
from typing import Literal

CONFIG_DIR = Path(__file__).parent / "configs"


def load_config(type: Literal["server", "web"]):
    config_file = CONFIG_DIR / f"{type}.yaml"
    try:
        with config_file.open("r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Config file {config_file} not found")


@click.group()
def cli():
    pass


@cli.command()
def run_server():
    args = load_config("server")
    app = FastAPI()

    # 添加跨域中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )


try:
    df = pd.read_csv(DATA_PATH)
    predictor = SimplePredictor(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Initialization failed: {e}")

# 字段映射关系
FIELD_MAP = {
    "Absorption": "Absorption max (nm)",
    "Emission": "Emission max (nm)",
    "Quantum Yield": "Quantum yield",
    "Stokes Shift": None,  # 需要模型预测
}


def get_predictions(smiles: str) -> dict:
    """使用模型进行预测"""
    try:
        predictions = predictor.predict([smiles])[0]
        return {
            "Absorption": predictions[0],
            "Emission": predictions[1],
            "Stokes Shift": predictions[2],
            "Quantum Yield": predictions[3],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/predict")
async def predict_properties(smiles: str):
    """
    通过 SMILES 查询属性，如果未找到则进行预测
    """
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES input is required.")

    # 查找是否已有记录
    matched_row = df[df["Chromophore"] == smiles]

    if not matched_row.empty:
        row = matched_row.iloc[0].replace({np.nan: None}).to_dict()
        response_data = {}

        # 填充数据，如果缺失则需要预测
        needs_prediction = False
        for output_field, csv_field in FIELD_MAP.items():
            value = row.get(csv_field) if csv_field else None
            if value is not None:
                response_data[output_field] = value
            else:
                response_data[output_field] = "predict"
                needs_prediction = True

        # 如果有字段需要预测
        if needs_prediction:
            pred_data = get_predictions(smiles)
            for field in response_data:
                if response_data[field] == "predict":
                    response_data[field] = pred_data[field]

        return {"status": "found", "data": response_data}

    else:
        # 完全找不到，全部用模型预测
        pred_data = get_predictions(smiles)
        return {"status": "not_found", "data": pred_data}


if __name__ == "__main__":
    predict_properties()
