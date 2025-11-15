
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

@st.cache_data
def cargar_base():
    return pd.read_csv("base_mercado_qro.csv")

df = cargar_base()

# --- Modelo ---
feature_cols = ["m2","recamaras","banos","estacionamientos",
                "precio_venta_mxn","antiguedad_anios","vacancia_pct","riesgo_zona"]
X = df[feature_cols]
y = df["renta_mensual_mxn"]

model = RandomForestRegressor(n_estimators=400,random_state=42,max_depth=10)
model.fit(X,y)

st.set_page_config(page_title="AI Deal Screener – Querétaro", layout="wide")
st.title("🏙️ AI Deal Screener – Querétaro")

st.caption("Herramienta interna para análisis de deals residenciales mediante IA/ML.")

st.sidebar.header("Parámetros del Deal")
zona = st.sidebar.selectbox("Zona", sorted(df["zona"].unique()))
m2 = st.sidebar.slider("m²", 40, 260, 110, 5)
recamaras = st.sidebar.slider("Recámaras", 1, 5, 3)
banos = st.sidebar.slider("Baños", 1, 4, 2)
estacionamientos = st.sidebar.slider("Estacionamientos", 0, 4, 2)
precio_venta = st.sidebar.number_input("Precio (MXN)",800000,20000000,3800000,50000)
antig = st.sidebar.slider("Antigüedad",0,40,8)
vac = st.sidebar.slider("Vacancia (%)",0.0,20.0,5.0,0.5)
gop = st.sidebar.slider("Gastos operativos (%)",5.0,40.0,18.0,0.5)

ltv = st.sidebar.slider("LTV (%)",0,90,70,5)
tasa = st.sidebar.slider("Tasa (%)",5.0,16.0,10.5,0.1)
plazo = st.sidebar.slider("Plazo (años)",5,30,20)

riesgo_map = {"Zibatá":1,"Cumbres del Lago":2,"Juriquilla":2,"Jurica":2,
              "El Refugio":3,"Corregidora":3,"Mileno III":3,"Centro":4}
riesgo = riesgo_map.get(zona,3)

input_df = pd.DataFrame([{
    "m2":m2,"recamaras":recamaras,"banos":banos,"estacionamientos":estacionamientos,
    "precio_venta_mxn":precio_venta,"antiguedad_anios":antig,
    "vacancia_pct":vac,"riesgo_zona":riesgo
}])

renta_pred = model.predict(input_df)[0]

st.subheader("📊 Resultados")
st.metric("Renta estimada", f"${renta_pred:,.0f} MXN")
st.write("Base interna cargada:", df.head())
