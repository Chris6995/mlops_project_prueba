import sys
from src.data import load_raw_data_v2, transform_to_time_series, transform_to_features_and_target
from src.paths import TRANSFORMED_DATA_DIR, PROCESSED_DATA_DIR

def main(year_month: str):
    # Parsea argumento YYYY_MM
    try:
        year_str, month_str = year_month.split('_')
        year, month = int(year_str), int(month_str)
    except ValueError:
        raise ValueError("El argumento debe tener formato YYYY_MM, e.g. 2025_04")

    # Carga y validación de datos crudos
    df_raw = load_raw_data_v2(year, month)
    if df_raw.empty:
        print(f"[FEATURE] No hay datos disponibles para {year_month}. Saliendo sin acciones.")
        sys.exit(0)

    # Transformación a serie temporal
    ts = transform_to_time_series(df_raw)
    # Generación de features y target
    X, y, df_full = transform_to_features_and_target(ts, location_id=None, n_lags=24)

    # Asegura existencia de directorios
    TRANSFORMED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Guarda outputs
    df_full.to_parquet(TRANSFORMED_DATA_DIR / "tabular_data.parquet", index=False)
    X.to_parquet(PROCESSED_DATA_DIR / "X.parquet", index=False)
    y.to_frame(name='target').to_parquet(PROCESSED_DATA_DIR / "y.parquet", index=False)

    print(f"[FEATURE] Pipeline completado para {year_month}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python feature_pipeline.py YYYY_MM")
        sys.exit(1)
    main(sys.argv[1])