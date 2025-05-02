import tempfile
import gradio as gr
import requests
import pandas as pd
import time
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors


def get_molecular_formula(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        formula = Chem.rdMolDescriptors.CalcMolFormula(molecule)
        return formula, molecule
    else:
        return "Invalid SMILES", None


def search_smiles(smiles):
    for i in range(5):
        time.sleep(0.3)
        yield f"Processing {i*20}%", None, None

    try:
        formula, molecule = get_molecular_formula(smiles)

        if molecule:
            img = Draw.MolToImage(molecule)
        else:
            img = None

        response = requests.get(
            "http://localhost:8000/predict", params={"smiles": smiles}, timeout=10
        )
        response.raise_for_status()

        data = response.json()
        result_df = pd.DataFrame([data["data"]])
        result_df["Molecular Formula"] = formula

        yield f"Done ✅ - Molecular Formula: {formula}", result_df, img

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        yield error_msg, None, None


def download_results(results_df):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", newline="", suffix=".csv"
    ) as temp_file:
        results_df.to_csv(temp_file, index=False)
        return temp_file.name


with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("🏠 Home"):
            gr.Markdown(
                """
            # 🔬 DyeLeS - Fluorescent Dye-Likeness Scoring Tool

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
                smiles_input = gr.Textbox(
                    label="Enter SMILES", placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O"
                )
                run_button = gr.Button("Submit")

            progress = gr.Textbox(label="Progress", interactive=False)
            table_output = gr.Dataframe(label="Results", type="pandas")
            molecule_image = gr.Image(label="Molecule Image", type="pil")
            download_button = gr.Button("Download Results", visible=False)

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


if __name__ == "__main__":
    demo.launch()
