import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from datetime import datetime
from google.cloud import storage
from river import linear_model, preprocessing, metrics

# =========================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# =========================================================
st.set_page_config(page_title="Aprendizaje en línea con River", page_icon="")
st.title("Aprendizaje en línea con River (Streaming realista desde Cloud Storage)")

st.markdown("""
Este panel demuestra cómo un modelo de **aprendizaje incremental** puede entrenarse y actualizarse 
a partir de un dataset grande alojado en **Google Cloud Storage (GCS)**.  
Cada archivo CSV del bucket se procesa como un *fragmento temporal* del flujo de datos.
""")

# =========================================================
# FUNCIONES AUXILIARES PARA GUARDAR Y CARGAR EL MODELO
# =========================================================
def save_model_to_gcs(model, bucket_name, destination_blob):
    """Guarda el modelo en formato pickle dentro del bucket de GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_string(pickle.dumps(model))
        st.success(f"Modelo guardado en GCS: `{destination_blob}`")
    except Exception as e:
        st.warning(f"No se pudo guardar el modelo: {e}")

def load_model_from_gcs(bucket_name, source_blob):
    """Carga el modelo desde GCS si existe, de lo contrario devuelve None."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)
        if blob.exists():
            data = blob.download_as_bytes()
            st.info("Modelo cargado desde GCS.")
            return pickle.loads(data)
        else:
            st.info("No se encontró un modelo previo, se iniciará uno nuevo.")
            return None
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo previo: {e}")
        return None

# =========================================================
# CONFIGURACIÓN DE PARÁMETROS
# =========================================================
bucket_name = st.text_input("Nombre del bucket de GCS:", "bucket_131025")
prefix = st.text_input("Carpeta/prefijo dentro del bucket:", "tlc_yellow_trips_2022/")
limite = st.number_input("Número de registros por archivo a procesar:", value=1000, step=100)
mostrar_grafico = st.checkbox("Mostrar gráfico de evolución del R²", value=True)

# =========================================================
# INICIALIZACIÓN DEL MODELO Y LAS MÉTRICAS
# =========================================================
MODEL_PATH = "models/model_incremental.pkl"

if "model" not in st.session_state:
    model = load_model_from_gcs(bucket_name, MODEL_PATH)
    if model is None:
        model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    st.session_state.model = model
    st.session_state.metric = metrics.R2()
    st.session_state.history = []  # Guarda evolución del R²

model = st.session_state.model
r2 = st.session_state.metric

# =========================================================
# EXTRACCIÓN DE PREDICTORES (ROBUSTA A COLUMNAS FALTANTES)
# =========================================================
def parse_pickup_dt(row):
    # 1) columna pre-derivada si existe
    if "pickup_hour" in row and pd.notna(row["pickup_hour"]):
        try:
            # puede venir como entero 0-23
            return None, int(row["pickup_hour"])
        except Exception:
            pass

    # 2) intenta con timestamps comunes en TLC
    for col in ("tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"):
        if col in row and pd.notna(row[col]):
            try:
                dt = pd.to_datetime(row[col], errors="coerce", utc=False)
                if pd.notna(dt):
                    return dt, int(dt.hour)
            except Exception:
                continue
    return None, 0

def extract_features(row):
    # Distancia
    dist = row.get("trip_distance", 0)
    try:
        dist = float(dist) if pd.notna(dist) else 0.0
    except Exception:
        dist = 0.0
    # Pasajeros
    psg = row.get("passenger_count", 0)
    try:
        psg = float(psg) if pd.notna(psg) else 0.0
    except Exception:
        psg = 0.0

    dt, hour = parse_pickup_dt(row)
    dow = dt.weekday() if isinstance(dt, pd.Timestamp) else 0
    is_weekend = 1.0 if dow >= 5 else 0.0

    x = {
        "dist": dist,
        "log_dist": np.log1p(dist),
        "pass": psg,
        "hour": float(hour),
        "dow": float(dow),
        "is_weekend": is_weekend,
    }
    return x

def valid_target(val):
    try:
        y = float(val)
        if not np.isfinite(y):
            return None
        return y
    except Exception:
        return None

# =========================================================
# FUNCIÓN DE STREAMING DESDE GCS
# =========================================================
def stream_from_bucket(bucket_name, prefix, limite=1000, chunksize=500):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    st.info(f"Se encontraron {len(blobs)} archivos en `{prefix}`.")

    for idx, blob in enumerate(blobs, start=1):
        st.write(f"Procesando archivo {idx} de {len(blobs)}: `{blob.name.split('/')[-1]}`")

        try:
            content = blob.download_as_bytes()
            buffer = io.BytesIO(content)

            count = 0
            for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):
                # iteración fila a fila; cada fila puede tener distinto esquema
                for _, row in chunk.iterrows():
                    if count >= limite:
                        break

                    y = valid_target(row.get("fare_amount"))
                    if y is None:
                        continue

                    x = extract_features(row)

                    # reglas simples anti-basura
                    if x["dist"] < 0 or x["dist"] > 200:
                        continue

                    y_pred = model.predict_one(x)
                    model.learn_one(x, y)
                    r2.update(y, y_pred)
                    count += 1

                if count >= limite:
                    break

        except Exception as e:
            st.warning(f"Error al procesar `{blob.name}`: {e}")
            continue

        yield blob.name, r2.get()

# =========================================================
# BOTÓN DE ACTUALIZACIÓN DEL MODELO
# =========================================================
if st.button("Actualizar modelo con datos del bucket"):
    st.info("Procesando archivos desde el bucket...")

    progreso = st.progress(0)
    nombres, valores = [], []

    blobs = list(storage.Client().bucket(bucket_name).list_blobs(prefix=prefix))
    total = max(len(blobs), 1)
    for i, (fname, score) in enumerate(stream_from_bucket(bucket_name, prefix, limite)):
        nombres.append(fname.split("/")[-1])
        valores.append(score)
        st.session_state.history.append(score)
        progreso.progress(min((i + 1) / total, 1.0))
        st.write(f"{fname} — R² acumulado: **{score:.3f}**")

    progreso.empty()
    st.success("Entrenamiento incremental completado.")
    save_model_to_gcs(model, bucket_name, MODEL_PATH)

    if mostrar_grafico and valores:
        st.line_chart(
            pd.DataFrame({"R²": valores}, index=np.arange(1, len(valores) + 1)),
            height=300,
            use_container_width=True
        )

# =========================================================
# SECCIÓN FINAL: ESTADO ACTUAL DEL MODELO
# =========================================================
st.markdown("---")
st.subheader("Estado actual del modelo")
st.write(f"R² actual: **{r2.get():.3f}**")

if st.session_state.history:
    st.line_chart(st.session_state.history, height=200, use_container_width=True)

st.caption("Cloud Run + River • Dataset público de taxis NYC (2022)")
