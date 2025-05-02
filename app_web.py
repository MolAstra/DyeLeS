from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import uvicorn
import gradio as gr
from dyeles import DyeLeS
from pathlib import Path
import yaml
from typing import Literal
from loguru import logger

from webserver_utils import download_results, load_config, predict_smiles


# 不再使用 click
def run_web():
    # 加载 web 配置
    args = load_config("web")

    # 设置 Gradio UI
    with gr.Blocks() as iface:
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
                            placeholder="e.g., N#Cc1cc2ccc(O)cc2oc1=O",
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
                    fn=lambda: "N#Cc1cc2ccc(O)cc2oc1=O",
                    inputs=[],
                    outputs=smiles_input,
                )

                run_button.click(
                    fn=predict_smiles,
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
                    outputs=gr.File(label="Download Results"),
                )

    # 启动 Gradio 应用
    iface.launch(
        server_name=args.get("host", "0.0.0.0"),
        server_port=int(args.get("port", 7860)),
        show_error=True,
    )


if __name__ == "__main__":
    run_web()
