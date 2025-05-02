from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
import uvicorn
from rdkit import Chem
from loguru import logger
from pathlib import Path

from webserver_utils import load_config, Predictor, clean_json


# 1. 定义请求模型，验证传入的数据
class PredictRequest(BaseModel):
    smiles: str  # 需要传入 SMILES 字符串


# 创建 FastAPI 应用
app = FastAPI(title="DyeLeS API", description="API for DyeLeS")

# 添加跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载配置
args = load_config("server")
predictor = Predictor(args["data_path"])


# 预测 API 路由
@app.post("/predict")
async def predict_properties(request: PredictRequest):
    smiles = request.smiles
    try:
        prediction = predictor.predict(smiles)
        prediction.update({"status": "success"})
        return clean_json(prediction)
    except Exception as e:
        logger.error(f"Error predicting for SMILES {smiles}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


# 启动 uvicorn 服务器
if __name__ == "__main__":
    uvicorn.run(
        "app_api:app",  # 使用模块路径而不是实例
        host=args.get("host", "0.0.0.0"),
        port=int(args.get("port", 8000)),
        reload=True,  # 启用开发模式的自动重载
    )
