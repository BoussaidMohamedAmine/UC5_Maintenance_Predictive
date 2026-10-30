# Jalon 4 — deployable model

This folder contains the saved Jalon 4 model converted to a loadable Keras file,
plus a self-contained Python 3.11 virtual environment and a minimal smoke test.

## Contents

| File | Purpose |
|---|---|
| `jalon4_final.keras` | Keras archive converted from `../models/jalon4_final.keras.zip` |
| `requirements.txt` | Pinned Python dependencies (CPU-only TensorFlow) |
| `.venv/` | Python 3.11 virtual environment with the requirements installed |
| `load_model.py` | Loads the model and runs a dummy prediction |
| `README.md` | This file |

## Model

- Architecture: `Masking` → `LSTM(32)` → `Dense(1, sigmoid)`
- Input: `(batch, 60, 20)` = 60 cycles × 20 channels
  - 10 values per cycle: 8 normalized sensors + scaled charge + health indicator (HI)
  - 10 first derivatives (one per value)
- Output: normalized RUL in `[0, 1]`; multiply by **125** to get cycles
- Trainable parameters: **6 817** (LSTM 6 784 + Dense 33)
- Total parameters: 20 453 (including optimizer state in the saved file)

The post-processing rule from the notebook (`RUL < 45` and `HI > 2` → force
`29` cycles) is **not** part of the Keras model; apply it in the calling code
if needed.

## Use the existing environment

```bash
cd deploy
.venv/bin/python load_model.py
```

Expected result: the model loads, prints its summary, and returns a dummy
prediction around `0.51` (normalized RUL).

## Recreate the environment from scratch

```bash
cd deploy
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

or with standard `venv`:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Preprocessing reminder

To match training:

1. Normalize the 8 sensors **per load regime** using the healthy training
   cycles (`RUL > 150`) of fleet A.
2. Scale the charge as `(charge - 80) / 20`.
3. Compute the health indicator with the Jalon 2 autoencoder
   (`../models/jalon2_autoencodeur.keras`), standardize its log-reconstruction
   error on healthy training cycles.
4. Build a 60-cycle window, prepend zeros for missing cycles, and append the
   first derivative of each of the 10 values.
5. Feed the `(60, 20)` array to the model and multiply the output by 125.

The exact pipeline is in `../jalon4_final.ipynb`.
