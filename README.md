# DyeLeS: Fluorescent Dye-Likeness Score and Its Application in Prioritizing Compound Libraries

## Setups

```bash
mamba create -n DyeLeS python=3.10
mamba activate DyeLeS
pip install loguru rdkit-pypi seaborn matplotlib pandas scikit-learn ipykernel absl-py tqdm "numpy<2"
pip install -e .
```

## Preprocess Data

```bash
python scripts/pre_dyes.py
python scripts/pre_coconut.py
python scripts/pre_zinc.py
```

- Visualize the processed data: `notebooks/visualize_processed_data.ipynb`

## Train Model

```bash
python scripts/train_model.py \
    --dye_csv data/processed/dyes.csv \
    --non_dye_csv data/processed/coconut.csv \
    --out_model_file dyles/resources/dye.model.gz
```

## Use DyeLeS

### Command Line

```bash
dyeles-score --input sample.csv --output sample_output.csv
```

### Python

- Detailed usage: `notebooks/tutorials.ipynb`

```python
from dyeles import DyeLeS

smiles = "C(=C/c1ccc(N(c2ccccc2)c2ccccc2)cc1)\c1ccc(-c2ccc(/C=C/c3ccc(N(c4ccccc4)c4ccccc4)cc3)cc2)cc1"
scorer = DyeLeS()
score = scorer.score(smiles)
print(f"score: {score}")
```
