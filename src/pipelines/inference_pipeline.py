import joblib
import pandas as pd
from pathlib import Path
from src.paths import PROCESSED_DATA_DIR, MODELS_DIR


def main():
    PROCESSED = Path(PROCESSED_DATA_DIR)
    MODELS = Path(MODELS_DIR)

    # Carga modelo y features procesados
    model = joblib.load(MODELS / 'linear_regression.pkl')
    X = pd.read_parquet(PROCESSED / 'X.parquet')

    # Genera predicciones
    preds = model.predict(X)
    df_pred = X.copy()
    df_pred['demand_pred'] = preds

    # Guarda predicciones
    df_pred.to_parquet(PROCESSED / 'predictions.parquet', index=False)
    print(f"[INFER] Predicciones guardadas en {PROCESSED / 'predictions.parquet'}")


if __name__ == '__main__':
    main()


