import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------
# 1. Descargar dataset
# ---------------------------------------------------
def download_data(year: int, month: int, output_path: str):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    response = requests.get(url)
    response.raise_for_status()

    output_file = Path(output_path) / f"rides_{year}_{month:02d}.parquet"
    with open(output_file, 'wb') as f:
        f.write(response.content)
    print(f"Archivo guardado en {output_file}")

# ---------------------------------------------------
# 2. Cargar y validar
# ---------------------------------------------------
def load_and_validate_data(file_path: str):
    df = pd.read_parquet(file_path)
    rides = df[['tpep_pickup_datetime', 'PULocationID']].copy()
    rides.rename(columns={'tpep_pickup_datetime': 'pickup_datetime', 'PULocationID': 'pickup_location_id'}, inplace=True)
    rides = rides[rides.pickup_datetime >= f'{rides.pickup_datetime.dt.year.min()}-01-01']
    rides = rides[rides.pickup_datetime < f'{rides.pickup_datetime.dt.year.max() + 1}-01-01']
    return rides

# ---------------------------------------------------
# 3. Agrupar por hora y localización
# ---------------------------------------------------
def aggregate_rides(rides: pd.DataFrame):
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    agg = rides.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index(name='rides')
    return agg

# ---------------------------------------------------
# 4. Completar huecos de fechas/horas
# ---------------------------------------------------
def add_missing_slots(agg_rides: pd.DataFrame):
    location_ids = agg_rides['pickup_location_id'].unique()
    full_range = pd.date_range(agg_rides['pickup_hour'].min(), agg_rides['pickup_hour'].max(), freq='H')

    output = []
    for location_id in tqdm(location_ids):
        df_loc = agg_rides[agg_rides.pickup_location_id == location_id]
        df_loc = df_loc.set_index('pickup_hour').reindex(full_range, fill_value=0).rename_axis('pickup_hour').reset_index()
        df_loc['pickup_location_id'] = location_id
        output.append(df_loc)
    
    return pd.concat(output)

# ---------------------------------------------------
# 5. Generar lags
# ---------------------------------------------------
def create_lag_features(df: pd.DataFrame, location_id: int, n_lags: int = 24):
    df_location = df[df.pickup_location_id == location_id].sort_values('pickup_hour').reset_index(drop=True)
    for lag in range(1, n_lags + 1):
        df_location[f'rides_previous_{lag}_hour'] = df_location['rides'].shift(lag)
    df_location.dropna(inplace=True)
    return df_location

# ---------------------------------------------------
# 6. Separar features y target
# ---------------------------------------------------
def split_features_target(df_location: pd.DataFrame, n_lags: int = 24):
    feature_cols = [f'rides_previous_{lag}_hour' for lag in range(1, n_lags + 1)]
    X = df_location[feature_cols]
    y = df_location['rides']
    return X, y
