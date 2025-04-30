import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.paths import PROCESSED_DATA_DIR, MODELS_DIR
from src.model import train_lightgbm, eval_model


def main():
    # Carga de datos procesados
    PROCESSED = Path(PROCESSED_DATA_DIR)
    MODELS = Path(MODELS_DIR)
    MODELS.mkdir(parents=True, exist_ok=True)

    X = pd.read_parquet(PROCESSED / 'X.parquet')
    y = pd.read_parquet(PROCESSED / 'y.parquet')['target']

    # Split train/val (manteniendo orden temporal)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    # Entrenamiento y evaluación
    model = train_lightgbm(X_train, y_train)
    metrics = eval_model(model, X_val, y_val)
    print(f"[TRAIN] Métricas de validación: {metrics}")

    # Guarda el modelo
    joblib.dump(model, MODELS / 'linear_regression.pkl')
    print(f"[TRAIN] Modelo guardado en {MODELS/'linear_regression.pkl'}")

if __name__ == '__main__':
    main()