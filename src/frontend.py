import zipfile 
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd

# plotting libraries
import streamlit as st
import geopandas as gpd
import pydeck as pdk

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference import (
    load_batch_of_features_from_store,
    load_model_from_registry,
    get_model_predictions
)
from src.paths import DATA_DIR
from src.plot import plot_one_sample

st.set_page_config(layout='wide')

# title
# current_date = pd.to_datetime(datetime.utcnow()).floor('H')
current_date = pd.Timestamp("2024-10-10 08:00:00")
st.title('Taxi Demand Prediction')
st.header(f'{current_date}')

progress_bar = st.sidebar.header('Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 7

def load_shape_data_file():
    """ Conocer zona del taxi a partir de las coordenadas de pickup"""
    # descargar el archivo zip con las zonas de taxi
    URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    response = requests.get(URL)
    path = DATA_DIR / f'taxi_zones.zip'
    if response.status_code == 200:
        open(path, "wb").write(response.content)
    else:
        raise Exception(f'{URL} is not available')
    
    # unzip file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')

    # cargamos y devolvemos las zonas de taxi
    return gpd.read_file(DATA_DIR / 'taxi_zones/taxi_zones.shp').to_crs('epsg:4326')


with st.spinner(text="Descargando localizaciones para graficar zonas"):
    geo_df = load_shape_data_file()
    st.sidebar.write('✅ Descargadas zonas de Taxi')
    progress_bar.progress(1 / N_STEPS)

with st.spinner(text="Obtención de funciones del almacén de funciones"):
    # load features from the feature store
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write('✅ Features de inferencia obtenidos')
    progress_bar.progress(2 / N_STEPS)
    print(f'{features}')

with st.spinner(text="Cargando modelo de ML del registro de modelos"):
    # load model from the model registry
    model = load_model_from_registry()
    st.sidebar.write('✅ Modelo de ML cargado')
    progress_bar.progress(3 / N_STEPS)

with st.spinner(text="Realizando predicciones"):
    # get predictions
    predictions = get_model_predictions(model, features)
    st.sidebar.write('✅ Predicciones obtenidas')
    progress_bar.progress(4 / N_STEPS)


with st.spinner(text="Preparing data to plot"):
    # prepare data to plot
    def pseudocolor(val, minval, maxval, startcolor, stopcolor):
        """
        Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the the one returned are
        composed of a sequence of N component values.

        Credits to https://stackoverflow.com/a/10907855
        """
        f = float(val-minval) / (maxval-minval)
        return tuple(f*(b-a)+a for (a, b) in zip(startcolor, stopcolor))
        
    df = pd.merge(geo_df, predictions,
                  right_on='pickup_location_id',
                  left_on='LocationID',
                  how='inner')
    
    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    progress_bar.progress(5/N_STEPS)

with st.spinner(text="Generating NYC Map"):
    # Crear un mapa de Nueva York con pydeck
    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=40.7831,
        longitude=-73.9712,
        zoom=11,
        max_zoom=16,
        pitch=45,
        bearing=0
    )
    # Crear un GeoJsonLayer para visualizar las zonas del mapa en las zonas con la demanda predicha
    geojson = pdk.Layer(
        "GeoJsonLayer",
        df,
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_elevation=10,
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        auto_highlight=True,
        pickable=True,
    )
    # Tooltip para mostrar la información de la zona y la demanda predicha
    tooltip = {"html": "<b>Zone:</b> [{LocationID}]{zone} <br /> <b>Predicted rides:</b> {predicted_demand}"}
    # Crear el objeto Deck de pydeck
    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )
    # Mostrar el mapa en Streamlit
    st.pydeck_chart(r)
    progress_bar.progress(6/N_STEPS)

    with st.spinner(text="Graficamos datos de series temporales"):
        
        row_indices = np.argsort(predictions['predicted_demand'].values)[-5:]  # top 5 predictions
        n_to_plot = 5

        # plot the top 5 predictions
        for row_id in row_indices[:n_to_plot]:
            fig = plot_one_sample(
                features=features,
                targets = predictions['predicted_demand'],
                example_id=row_id,
                predictions = pd.Series(predictions['predicted_demand'])
            )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

        progress_bar.progress(6/N_STEPS)