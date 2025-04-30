from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    FeatureService,
    on_demand_feature_view,
)
from feast.value_type import ValueType  # Importa ValueType para value_type
from feast.types import Int64  # Importa Int64 para dtype
from datetime import timedelta
import pandas as pd


# 1) Define la entidad (clave primaria)
pickup_loc = Entity(
    name="pickup_location_id",
    join_keys=["pickup_location_id"],
    value_type=ValueType.INT64,  # Usa ValueType.INT64 para evitar errores
)

# 2) Apunta al origen de los datos históricos
rides_source = FileSource(
    name="rides_hourly_source",
    path="data/driver_stats.parquet",
    timestamp_field="pickup_datetime",
)

# 3) Define el FeatureView que materializa esa tabla
rides_fv = FeatureView(
    name="rides_hourly",
    entities=[pickup_loc],
    ttl=timedelta(days=1),
    schema=[Field(name="rides", dtype=Int64)],  # Usa Int64 para dtype
    online=True,
    source=rides_source,
)


# 4) Calcula manualmente la característica bajo demanda
def avg_last_24h(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["rides_last_24h_avg"] = inputs["rides"].rolling(24).mean().fillna(0).astype(int)
    return df

# 5) Agrupa tus FV’s en un FeatureService
rides_service = FeatureService(
    name="rides_service_v1",
    features=[rides_fv],  # Pasa el FeatureView completo
)