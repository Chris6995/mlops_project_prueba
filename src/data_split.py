from datetime import datetime
from typing import Tuple
import pandas as pd


def train_test_split(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column_name: str = "target",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Divide un DataFrame de series temporales en train y test según una fecha de corte.

    Args:
        df (pd.DataFrame): DataFrame completo, debe incluir columna 'pickup_hour' y la columna target.
        cutoff_date (datetime): Fecha usada como límite para dividir entre train y test.
        target_column_name (str): Nombre de la columna objetivo (target).

    Returns:
        X_train, y_train, X_test, y_test: Datos de entrenamiento y prueba separados.
    """
    # División temporal train-test
    train_data = df[df.pickup_hour < cutoff_date].reset_index(drop=True)
    test_data = df[df.pickup_hour >= cutoff_date].reset_index(drop=True)

    # Separar features (X) y target (y)
    X_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]
    X_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]

    return X_train, y_train, X_test, y_test
