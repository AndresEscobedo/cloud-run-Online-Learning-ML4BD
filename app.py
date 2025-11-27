import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from google.cloud import storage
from river import linear_model, preprocessing, metrics

# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(page_title="Aprendizaje en línea", page_icon="")
st.title("Aprendizaje en línea con River (Step-by-step desde GCS)")

st.markdown("""
Este panel replica la lógica del entrenamiento incremental,
pero procesa **un archivo por clic**, con limpieza robusta y control de outliers.
""")

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def save_model_to_gcs(model, bucket_name, destination_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_string(pickle.dumps(model))
        st.success(f"Modelo guardado en GCS: {destination_blob}")
    except Exception as e:
        st.warning(f"No se pudo guardar el modelo: {e}")

def load_model_from_gcs(bucket_name, source_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)

        if blob.exists():
            data = blob.download_as_bytes()
            st.info("Modelo cargado desde GCS.")
            return pickle.loads(data)
        return None

    except Exception as e:
        st.warning(f"No se pudo cargar el modelo previo: {e}")
        return None

# =========================================================
# PARÁMETROS
# =========================================================
bucket_name = st.text_input("Bucket de GCS:", "mlbd_bucket_131025")
prefix = st.text_input("Prefijo/carpeta:", "tlc_yellow_trips_2022/")
limite = st.number_input("Filas a procesar por archivo:", value=1000, step=100)

MODEL_PATH = "models/model_incremental.pkl"

# Umbrales de limpieza y seguridad
MIN_ROWS_PER_CHUNK = 50        # mínimo de filas limpias para usar un chunk
MAX_R2_ABS = 1e6               # guard-rail para R² patológico (no debería llegar aquí con la limpieza)
COST_PER_MILE_MIN = 0.5        # tarifa mínima razonable por milla
COST_PER_MILE_MAX = 20.0       # tarifa máxima razonable por milla

# =========================================================
# INICIALIZAR MODELO
# =========================================================
if "model" not in st.session_state:
    model = load_model_from_gcs(bucket_name, MODEL_PATH)
    if model is None:
        model = preprocessing.StandardScaler() | linear_model.LinearRegression()

    st.session_state.model = model
    st.session_state.metric = metrics.R2()
    st.session_state.history = []

    st.session_state.blobs = None
    st.session_state.index = 0
    st.session_state._last_bucket = bucket_name
    st.session_state._last_prefix = prefix

# resetear cache si cambian bucket/prefix
if (st.session_state._last_bucket != bucket_name) or (st.session_state._last_prefix != prefix):
    st.session_state.blobs = None
    st.session_state.index = 0
    st.session_state._last_bucket = bucket_name
    st.session_state._last_prefix = prefix

model = st.session_state.model
metric = st.session_state.metric

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def _parse_time_fields(row):
    if "pickup_hour" in row and pd.notna(row["pickup_hour"]):
        try:
            hour = int(pd.to_numeric(row["pickup_hour"], errors="coerce"))
            return None, max(0, min(hour, 23))
        except Exception:
            pass

    for c in ("tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"):
        if c in row and pd.notna(row[c]):
            dt = pd.to_datetime(row[c], errors="coerce", utc=False)
            if pd.notna(dt):
                return dt, int(dt.hour)
    return None, 0

def _safe_float(x, default=0.0):
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v) or not np.isfinite(v):
        return float(default)
    return float(v)

def _extract_x(row):
    dist = _safe_float(row.get("trip_distance", 0.0), 0.0)
    psg  = _safe_float(row.get("passenger_count", 0.0), 0.0)

    dt, hour = _parse_time_fields(row)
    dow = int(dt.weekday()) if isinstance(dt, pd.Timestamp) else 0
    weekend = 1.0 if dow >= 5 else 0.0

    x = {
        "dist": dist,
        "log_dist": float(np.log1p(max(dist, 0.0))),
        "pass": psg,
        "hour": float(hour),
        "dow": float(dow),
        "is_weekend": weekend,
    }

    # comprobar que ningún predictor es inf/NaN
    for v in x.values():
        if not np.isfinite(v):
            return None
    return x

def _valid_target(v):
    y = pd.to_numeric(v, errors="coerce")
    if pd.isna(y) or not np.isfinite(y):
        return None
    return float(y)

# =========================================================
# LIMPIEZA ROBUSTA DE CADA CHUNK
# =========================================================
def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # columnas mínimas requeridas
    required = {"trip_distance", "passenger_count", "fare_amount"}
    if not required.issubset(chunk.columns):
        return chunk.iloc[0:0]  # vacío

    # coerción numérica
    for col in ["trip_distance", "passenger_count", "fare_amount"]:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    # eliminar NaNs e infinitos en columnas críticas
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    chunk = chunk.dropna(subset=["trip_distance", "passenger_count", "fare_amount"])
    if chunk.empty:
        return chunk

    # filtros físicos duros amplios
    chunk = chunk[
        (chunk["fare_amount"] >= 0.0) &
        (chunk["trip_distance"] > 0.0) &
        (chunk["passenger_count"] >= 0.0)
    ]
    if chunk.empty:
        return chunk

    # recorte por cuantiles (protege contra colas extremas)
    q_dist_hi = float(chunk["trip_distance"].quantile(0.99))
    q_fare_hi = float(chunk["fare_amount"].quantile(0.99))

    max_dist = min(max(q_dist_hi, 1.0), 50.0)    # nunca más de 50 millas
    max_fare = min(max(q_fare_hi, 5.0), 300.0)   # nunca más de 300 USD

    # filtros básicos "razonables"
    chunk = chunk[
        (chunk["fare_amount"].between(2.0, max_fare)) &
        (chunk["trip_distance"].between(0.1, max_dist)) &
        (chunk["passenger_count"].between(1, 6))
    ]
    if chunk.empty:
        return chunk

    # filtro por tarifa por milla (evita combinaciones absurdas fare/dist)
    dist_clip = chunk["trip_distance"].clip(lower=0.1)
    cost_per_mile = chunk["fare_amount"] / dist_clip
    chunk = chunk[
        (cost_per_mile >= COST_PER_MILE_MIN) &
        (cost_per_mile <= COST_PER_MILE_MAX)
    ]
    if chunk.empty:
        return chunk

    # proteger contra chunks degenerados (casi sin variabilidad de Y)
    if chunk["fare_amount"].std() < 1.0:
        # chunk no aporta casi nada para R²; mejor saltarlo para la métrica
        return chunk.iloc[0:0]

    if len(chunk) < MIN_ROWS_PER_CHUNK:
        return chunk.iloc[0:0]

    return chunk

# =========================================================
# PROCESAR UN SOLO ARCHIVO
# =========================================================
def process_single_blob(bucket_name, blob_name, limite=1000, chunksize=500):
    global model, metric
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    try:
        content = blob.download_as_bytes()
        buffer = io.BytesIO(content)
        count = 0

        for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):
            # limpieza robusta centralizada
            chunk = _clean_chunk(chunk)
            if chunk.empty:
                continue

            for _, row in chunk.iterrows():
                if count >= limite:
                    break

                y = _valid_target(row["fare_amount"])
                if y is None:
                    continue

                x = _extract_x(row)
                if x is None:
                    continue

                # seguridad extra en distancia
                if x["dist"] < 0.0 or x["dist"] > 200.0:
                    continue

                # predicción y actualización
                try:
                    pred = model.predict_one(x)
                except Exception:
                    # si el modelo internamente revienta, saltar esta fila
                    continue

                try:
                    model.learn_one(x, y)
                except Exception:
                    # si el aprendizaje falla, no tocar la métrica
                    continue

                # actualizar métrica solo si pred y y son finitos
                if np.isfinite(pred) and np.isfinite(y):
                    metric.update(y, pred)

                count += 1

                # guard-rail contra R² patológico
                current_r2 = metric.get()
                if not np.isfinite(current_r2) or abs(current_r2) > MAX_R2_ABS:
                    st.warning(
                        f"R² fuera de rango razonable ({current_r2:.2e}) en {blob_name}. "
                        "Se detiene el procesamiento de este archivo y se reinicia la métrica."
                    )
                    metric = metrics.R2()
                    st.session_state.metric = metric
                    return metric.get()

            if count >= limite:
                break

    except Exception as e:
        st.warning(f"Error en {blob_name}: {e}")
        return None

    return metric.get()

# =========================================================
# BOTÓN: PROCESAR SIGUIENTE ARCHIVO
# =========================================================
if st.button("Procesar siguiente archivo"):

    if st.session_state.blobs is None:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        st.session_state.blobs = sorted(
            list(bucket.list_blobs(prefix=prefix)),
            key=lambda b: b.name
        )
        st.session_state.index = 0
        st.info(f"Se encontraron {len(st.session_state.blobs)} archivos.")

    blobs = st.session_state.blobs
    idx = st.session_state.index

    if idx >= len(blobs):
        st.success("Todos los archivos ya fueron procesados.")
    else:
        blob = blobs[idx]
        short = blob.name.split("/")[-1]
        st.write(f"Procesando {idx+1}/{len(blobs)}: `{short}`")

        score = process_single_blob(bucket_name, blob.name, int(limite))

        if score is not None:
            st.session_state.history.append(score)
            st.write(f"{blob.name} — R² acumulado: **{score:.3f}**")
            save_model_to_gcs(model, bucket_name, MODEL_PATH)

        st.session_state.index += 1

# =========================================================
# ESTADO FINAL
# =========================================================
st.markdown("---")
st.subheader("Estado actual del modelo")
st.write(f"R² actual: **{metric.get():.3f}**")

if st.session_state.history:
    st.line_chart(st.session_state.history)

st.caption("Cloud Run + River • Dataset público de taxis NYC (2022)_221125")
