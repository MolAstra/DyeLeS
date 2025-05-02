from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import uvicorn
import gradio as gr
from dyeles import DyeLeS
from pathlib import Path
import click
from rdkit import Chem
import yaml
from typing import Literal
from loguru import logger

from webserver_utils import search_smiles, download_results

CONFIG_DIR = Path(__file__).parent / "configs"


def load_config(type: Literal["server", "web"]):
    config_file = CONFIG_DIR / f"{type}.yaml"
    logger.info(f"Loading config from {config_file}")
    try:
        with config_file.open("r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Config file {config_file} not found")


class Predictor:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.db = self.load_db()
        self.model = self.load_model()

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

    def load_model(self):
        return DyeLeS()

    def get_canonical_smiles(self, smiles: str) -> str:
        """获取标准SMILES"""
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    def search_db(self, smiles: str) -> dict:
        """从数据库中查找是否已有记录"""
        canonical_smiles = self.get_canonical_smiles(smiles)
        if canonical_smiles is None:
            return None
        hit = self.db[self.db["smiles"] == canonical_smiles]
        if hit.empty:
            return None
        return hit.to_dict(orient="records")

    def predict(self, smiles: str) -> dict:
        """1. 从数据库中查找是否已有记录
        2. 如果已有记录，则返回记录中的数据
        3. 如果未找到，则使用模型预测
        """
        hit = self.search_db(smiles)
        if hit is not None:
            return hit
        else:
            return self.model(smiles)


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

    predictor = Predictor(args["data_path"])

    @app.get("/predict")
    async def predict_properties(request: Request):
        request_data = await request.json()
        smiles = request_data.get("smiles")
        return {"status": "success", "data": predictor.predict(smiles)}

    uvicorn.run(app, host=args.get("host", "0.0.0.0"), port=int(args.get("port", 8000)))


@cli.command()
def run_web():
    args = load_config("web")
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("🏠 Home"):
                gr.Markdown(
                    """# 🔬 DyeLeS - Fluorescent Dye-Likeness Scoring Tool

                    Welcome to **DyeLeS**, a web-based application designed for predicting molecular properties, 
                    especially focused on fluorescent dye-likeness.

                    ## 🔧 Features
                    - SMILES-based molecular property prediction
                    - Molecular formula display
                    - Molecule structure visualization
                    - Downloadable results

                    ## 📚 Use Case
                    Useful for:
                    - Computational chemists
                    - Drug discovery
                    - High-throughput compound screening

                    Navigate to the **Prediction** tab to get started!

                    ---

                    ### 📄 Citation:
                    If you use this tool in your research, please cite our work:
                    **"DyeLeS: A Web-Based Application and Database for Fluorescent Dye-Likeness Scoring and Its Application in Prioritizing Compound Libraries"**

                    ### 💻 GitHub:
                    Find the source code and contribute to the project at:
                    [GitHub Repository](https://github.com/MolAstra/DyeLeS)
                    """
                )

            with gr.TabItem("🧪 SMILES Property Prediction"):
                gr.Markdown("## 🧪 SMILES Property Prediction Demo")

                with gr.Row():
                    with gr.Column(scale=3):
                        smiles_input = gr.Textbox(
                            label="Enter SMILES",
                            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
                        )
                    with gr.Column(scale=1):
                        example_button = gr.Button("Use Example")
                        run_button = gr.Button("Submit")

                progress = gr.Textbox(label="Progress", interactive=False)
                table_output = gr.Dataframe(label="Results", type="pandas")
                molecule_image = gr.Image(label="Molecule Image", type="pil")
                download_button = gr.Button("Download Results", visible=False)

                # 功能绑定
                example_button.click(
                    fn=lambda: "CC(=O)OC1=CC=CC=C1C(=O)O",
                    inputs=[],
                    outputs=smiles_input,
                )

                run_button.click(
                    fn=search_smiles,
                    inputs=smiles_input,
                    outputs=[progress, table_output, molecule_image],
                )

                table_output.change(
                    fn=lambda df: (
                        gr.update(visible=True)
                        if df is not None
                        else gr.update(visible=False)
                    ),
                    inputs=table_output,
                    outputs=download_button,
                )

                download_button.click(
                    fn=download_results,
                    inputs=table_output,
                    outputs=gr.File(label="Download CSV"),
                )
    demo.launch(
        server_name=args.get("host", "0.0.0.0"),
        server_port=int(args.get("port", 7860)),
        show_error=True,
    )


cli.add_command(run_server)
cli.add_command(run_web)

if __name__ == "__main__":
    cli()
