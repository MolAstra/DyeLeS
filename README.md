# DyeLeS: Fluorescent Dye-Likeness Score and Its Application in Prioritizing Compound Libraries

## Setups

```bash
git clone https://github.com/zhaisilong/DyeLeS.git
cd DyeLeS

mamba create -n DyeLeS python=3.10
mamba activate DyeLeS

# If you want to reproduce the results in the paper
# git checkout v0.1.0

pip install -e .

mamba install -c tmap tmap
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
    --out_model_file dyeles/resources/dye.model.gz
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
score = scorer(smiles)
print(f"score: {score}")
```

### Webserver

```bash
python app_api.py  # API server, 8000 port
python app_web.py  # Web server, 8001 port
```
