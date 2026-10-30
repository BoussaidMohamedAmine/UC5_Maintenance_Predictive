"""Load and smoke-test jalon4_final.keras.

Usage:
    .venv/bin/python load_model.py
"""

from pathlib import Path

import numpy as np
import keras

MODEL_PATH = Path(__file__).with_name("jalon4_final.keras")


def main() -> None:
    model = keras.models.load_model(MODEL_PATH)
    model.summary()

    # The model expects (batch, 60 cycles, 20 channels):
    # 10 features (8 sensors + charge + health indicator) + their derivatives.
    dummy = np.zeros((1, 60, 20), dtype="float32")
    prediction = model.predict(dummy, verbose=0)
    print(f"\nLoaded: {MODEL_PATH}")
    print(f"Input shape : {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Prediction  : {float(np.ravel(prediction)[0]):.4f} (normalized RUL)")


if __name__ == "__main__":
    main()
