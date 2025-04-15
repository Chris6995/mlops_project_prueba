import pandas as pd
import requests
from tqdm import tqdm
from typing import Tuple
from typing import Optional, List
from paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------
# Descargar datos
# ---------------------------------------------------


def download_one_file_of_raw_data(year: int, month: int) -> Path:
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02}.parquet"
    output_path = RAW_DATA_DIR / f"rides_{year}_{month:02}.parquet"

    if not output_path.exists():
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Archivo descargado: {output_path}")
        else:
            raise Exception(f"No se pudo descargar el archivo desde {url}")
    else:
        print(f"El archivo ya existe: {output_path}")

    return output_path

# ---------------------------------------------------
# Cargar datos
# ---------------------------------------------------


def load_data(year: int, month: int) -> pd.DataFrame:
    path = RAW_DATA_DIR / f"rides_{year}_{month:02}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    return pd.read_parquet(path)

# ---------------------------------------------------
# Validar datos
# ---------------------------------------------------


def validate_data(df: pd.DataFrame, min_date="2024-01-01", max_date="2024-02-01") -> pd.DataFrame:
    df = df[['tpep_pickup_datetime', 'PULocationID']].copy()
    df = df.rename(columns={
        'tpep_pickup_datetime': 'pickup_datetime',
        'PULocationID': 'pickup_location_id'
    })
    df = df[(df['pickup_datetime'] >= min_date) & (df['pickup_datetime'] < max_date)]
    return df

# ---------------------------------------------------
# Cargar y validar datos
# ---------------------------------------------------


def load_raw_data(
    year: int,
    months: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Descarga, carga y valida los datos crudos de NYC Taxi para un año y meses específicos.
    """
    rides = pd.DataFrame()

    if months is None:
        months = list(range(1, 13))
    elif isinstance(months, int):
        months = [months]

    for month in months:
        local_file = RAW_DATA_DIR / f'rides_{year}_{month:02}.parquet'
        if not local_file.exists():
            try:
                print(f'Descargando archivo {year}-{month:02}')
                download_one_file_of_raw_data(year, month)
            except:
                print(f'{year}-{month:02} no está disponible')
                continue
        else:
            print(f'Archivo {year}-{month:02} ya está disponible localmente')

        rides_one_month = pd.read_parquet(local_file)
        rides_one_month = validate_data(rides_one_month)
        rides = pd.concat([rides, rides_one_month])

    if rides.empty:
        return pd.DataFrame()
    else:
        return rides[['pickup_datetime', 'pickup_location_id']]
    
# ---------------------------------------------------
# Fill missing values de fechas/horas
# ---------------------------------------------------

def add_missing_slots(agg_rides: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena las horas faltantes para cada localización con 0 viajes.

    Args:
        agg_rides (pd.DataFrame): DataFrame con columnas ['pickup_hour', 'pickup_location_id', 'rides']

    Returns:
        pd.DataFrame: DataFrame con huecos rellenados por hora y localización.
    """
    location_ids = agg_rides['pickup_location_id'].unique()
    full_range = pd.date_range(
        agg_rides['pickup_hour'].min(),
        agg_rides['pickup_hour'].max(),
        freq='H'
    )

    output = []

    for location_id in tqdm(location_ids, desc="Rellenando huecos por localización"):
        df_loc = agg_rides[agg_rides['pickup_location_id'] == location_id]
        df_loc = df_loc.set_index('pickup_hour').reindex(full_range, fill_value=0)
        df_loc['pickup_location_id'] = location_id
        df_loc.reset_index(inplace=True)
        df_loc.rename(columns={'index': 'pickup_hour'}, inplace=True)
        output.append(df_loc)

    return pd.concat(output, ignore_index=True)[['pickup_hour', 'pickup_location_id', 'rides']]

# ---------------------------------------------------
# Transformar datos a time series
# ---------------------------------------------------


def transform_to_time_series(rides: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma los datos en una serie temporal con número de viajes por hora y localización.
    Llama internamente a add_missing_slots para rellenar los huecos.

    Args:
        rides (pd.DataFrame): DataFrame con columnas ['pickup_datetime', 'pickup_location_id']

    Returns:
        pd.DataFrame: DataFrame con columnas ['pickup_hour', 'pickup_location_id', 'rides']
    """
    # Redondear a la hora más cercana
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')

    # Agrupar número de viajes por hora y localización
    agg_rides = (
        rides.groupby(['pickup_hour', 'pickup_location_id'])
        .size()
        .reset_index(name='rides')
    )

    agg_rides_all_slots = add_missing_slots(agg_rides)
    # Rellenar huecos horarios
    return agg_rides_all_slots

# ---------------------------------------------------
# Generar lags
# ---------------------------------------------------

def create_lag_features(df: pd.DataFrame, n_lags: int = 24) -> pd.DataFrame:
    """
    Crea n_lags features con retrasos (lags) para la serie temporal de rides.
    La columna target será el valor actual a predecir.

    Args:
        df (pd.DataFrame): DataFrame con columnas ['pickup_hour', 'pickup_location_id', 'rides']
        n_lags (int): número de lags que queremos generar.

    Returns:
        pd.DataFrame: dataframe con columnas rides_previous_N_hour y target
    """
    df = df.sort_values('pickup_hour').reset_index(drop=True)
    
    for lag in range(n_lags, 0, -1):
        df[f'rides_previous_{lag}_hour'] = df['rides'].shift(lag)
    
    df['target'] = df['rides']
    df = df.drop(columns=['rides'])
    df = df.dropna().reset_index(drop=True)
    return df

# ---------------------------------------------------
# Transformar time series a features y target
# ---------------------------------------------------


from typing import Optional, Tuple
import pandas as pd

def transform_to_features_and_target(
    ts_data: pd.DataFrame, 
    location_id: Optional[int] = None, 
    n_lags: int = 24
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepara los datos como features y target para entrenar un modelo.
    Si se especifica `location_id`, se filtran los datos por esa localización.

    Args:
        ts_data (pd.DataFrame): DataFrame con ['pickup_hour', 'pickup_location_id', 'rides']
        location_id (int, optional): ID de la zona a usar. Si None, se usan todos los datos.
        n_lags (int): número de lags (horas previas) a usar como features

    Returns:
        Tuple: X (features), y (target), df_location (datos procesados)
    """
    df_location = ts_data.copy()

    if location_id is not None:
        df_location = df_location[df_location.pickup_location_id == location_id].copy()

    # Aseguramos orden correcto antes de crear lags
    df_location = df_location.sort_values('pickup_hour').reset_index(drop=True)

    df_location = create_lag_features(df_location, n_lags=n_lags)

    X = df_location.drop(columns=['target', 'pickup_hour', 'pickup_location_id'])
    y = df_location['target']

    return X, y, df_location



# ---------------------------------------------------
# Cargar datos de un año
# ---------------------------------------------------

def load_last_12_months_data(end_date: datetime) -> pd.DataFrame:
    """
    Carga los datos de los 12 meses anteriores a una fecha dada (end_date),
    realiza la transformación de columnas y filtra por fechas válidas.
    
    Args:
        end_date (datetime): Fecha de corte (no incluida)
    
    Returns:
        pd.DataFrame: DataFrame combinado y validado con columnas:
                      ['pickup_datetime', 'pickup_location_id']
    """
    rides_all = pd.DataFrame()

    # Calcular el primer mes a cargar (12 meses atrás)
    start_date = end_date - relativedelta(months=12)

    current_date = start_date
    while current_date < end_date:
        year = current_date.year
        month = current_date.month

        print(f"Descargando datos de {year}-{month:02d}")
        try:
            # Descargar archivo
            download_one_file_of_raw_data(year, month)

            # Cargar parquet
            df_raw = pd.read_parquet(f"../data/raw/rides_{year}_{month:02}.parquet")
            print(f"Datos cargados desde ../data/raw/rides_{year}_{month:02}.parquet")

            # Seleccionar y renombrar columnas
            rides = df_raw[['tpep_pickup_datetime','PULocationID']].copy()
            rides.rename(columns={
                'tpep_pickup_datetime': 'pickup_datetime',
                'PULocationID': 'pickup_location_id'
            }, inplace=True)

            # Filtrar por fechas del mes actual
            month_start = datetime(year, month, 1)
            if month == 12:
                month_end = datetime(year + 1, 1, 1)
            else:
                month_end = datetime(year, month + 1, 1)
            rides = rides[(rides.pickup_datetime >= month_start) & 
                          (rides.pickup_datetime < month_end)]

            rides_all = pd.concat([rides_all, rides])

        except Exception as e:
            print(f"❌ Error en {year}-{month:02d}: {e}")

        current_date += relativedelta(months=1)

    rides_all = rides_all.sort_values("pickup_datetime").reset_index(drop=True)
    return rides_all