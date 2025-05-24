import hopsworks
import pandas as pd
import numpy as np
from datetime import timedelta

class Transformer:
    def __init__(self):
        # Se ejecuta al iniciar el contenedor
        project = hopsworks.login()  
        fs      = project.get_feature_store()
        # Usa tu Feature View de series pre-pivotadas
        self.fv = fs.get_feature_view(name="time_series_hourly_feature_view", version=1)
        self.N_FEATURES = 24  # o el N que entrenaste

    def transform_input(self, request: dict) -> dict:
        # request vendrá con tu JSON, p.ej.:
        # { "pickup_location_id": 42, "pickup_hour": "2024-05-23T12:00:00Z" }
        pid = request["pickup_location_id"]
        ph  = pd.Timestamp(request["pickup_hour"]).tz_convert("UTC")

        # Recupera raw en la ventana [ph-1d, ph-1h]
        start = ph - timedelta(days=1)
        end   = ph - timedelta(hours=1)
        raw = self.fv.get_batch_data(start_time=start, end_time=ph).to_df()

        # Filtra sólo tu location_id
        df = raw[raw["pickup_location_id"] == pid].sort_values("pickup_hour")

        # Construye tu vector de N_FEATURES
        vals = df["rides"].values
        last_n = vals[-self.N_FEATURES:]
        if len(last_n) < self.N_FEATURES:
            pad = np.zeros(self.N_FEATURES - len(last_n), dtype=np.float32)
            last_n = np.concatenate([pad, last_n])

        # Devuelve la forma que sklearnserver espera:
        # { "data": [[ feat1, feat2, … ]] }
        return { "data": [ last_n.tolist() ] }

    def transform_output(self, response: dict) -> dict:
        # response será { "predictions": [ 12.345 ] }
        pred = response["predictions"][0]
        return { "predicted_demand": int(round(pred)) }
