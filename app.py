import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from PIL import Image

# =========================
# 1. BASE DE DATOS DE MERCADO – QUERÉTARO
# =========================

@st.cache_data
def cargar_base():
    return pd.read_csv("base_mercado_qro.csv")

df = cargar_base()

# =========================
# 2. ENTRENAR MODELOS DE ML
# =========================

# Modelo de RENTAS: usa precio como feature y predice renta
feature_cols_rent = [
    "m2", "recamaras", "banos", "estacionamientos",
    "precio_venta_mxn", "antiguedad_anios", "vacancia_pct", "riesgo_zona"
]
X_rent = df[feature_cols_rent]
y_rent = df["renta_mensual_mxn"]

model_rent = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    max_depth=10
)
model_rent.fit(X_rent, y_rent)

# Modelo de PRECIO: NO usa precio como feature; predice precio_venta_mxn
feature_cols_price = [
    "m2", "recamaras", "banos", "estacionamientos",
    "antiguedad_anios", "vacancia_pct", "riesgo_zona"
]
X_price = df[feature_cols_price]
y_price = df["precio_venta_mxn"]

model_price = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    max_depth=10
)
model_price.fit(X_price, y_price)

# =========================
# 3. CONFIGURACIÓN DE PÁGINA
# =========================

st.set_page_config(
    page_title="AI Deal Screener – Querétaro",
    layout="wide"
)
# --- Ajuste para reducir el espacio entre logo y título ---
st.markdown("""
    <style>
        /* Reduce el padding superior general */
        .main {
            padding-top: 0rem !important;
        }

        /* Reduce espacio del contenedor del logo */
        .logo-container {
            margin-bottom: -60px !important;
        }

        /* Acerca el título al logo */
        h1 {
            margin-top: 50px !important;
        }
    </style>
""", unsafe_allow_html=True)
# --- Fin del ajuste ---
# Logo centrado en la parte superior (opcional)
try:
    logo = Image.open("logo.png")
    st.markdown("<div class='logo-container' style='text-align: center;'>", unsafe_allow_html=True)
    st.image(logo, width=220)
    st.markdown("</div>", unsafe_allow_html=True)
except Exception:
    pass

st.title("🏙️ Modelo de Pricing AI/ML – Querétaro")
st.caption("Herramienta interna de análisis de oportunidades residenciales con modelos de IA / ML")

st.markdown(
    """
    Este panel funciona como un **copiloto de análisis** para el mercado residencial de Querétaro.

    - **Módulo Rentas**: estima la renta de mercado y calcula NOI, cap rate, cash-on-cash y DSCR.
    - **Módulo Precio de venta**: estima el precio de venta recomendado para vivienda nueva o reciente.
    """
)

# =========================
# 4. SIDEBAR: SELECCIÓN DE MÓDULO E INPUTS
# =========================

st.sidebar.header("🧩 Módulo de análisis")
modulo = st.sidebar.radio(
    "Selecciona el módulo",
    ["Rentas", "Precio de venta"]
)

st.sidebar.header("🏠 Parámetros del inmueble")

zona = st.sidebar.selectbox(
    "Zona",
    sorted(df["zona"].unique())
)

m2 = st.sidebar.slider("Metros cuadrados construidos", 40, 260, 110, step=5)
recamaras = st.sidebar.slider("Recámaras", 1, 5, 3)
banos = st.sidebar.slider("Baños", 1, 4, 2)
estacionamientos = st.sidebar.slider("Estacionamientos", 0, 4, 2)

precio_venta_mxn = st.sidebar.number_input(
    "Precio de compra / precio objetivo (MXN)",
    min_value=800000,
    max_value=20000000,
    value=3800000,
    step=50000
)

if modulo == "Precio de venta":
    default_antiguedad = 0
else:
    default_antiguedad = 8

antiguedad_anios = st.sidebar.slider("Antigüedad (años)", 0, 40, default_antiguedad)
vacancia_pct = st.sidebar.slider("Vacancia estimada (%)", 0.0, 20.0, 5.0, step=0.5)

if modulo == "Rentas":
    gastos_operativos_pct = st.sidebar.slider(
        "Gastos operativos (% del ingreso bruto)",
        5.0, 40.0, 18.0, step=0.5
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Supuestos de financiamiento")
    ltv_pct = st.sidebar.slider("LTV (porcentaje financiado)", 0, 90, 70, step=5)
    tasa_interes_anual = st.sidebar.slider("Tasa de interés anual (%)", 5.0, 16.0, 10.5, step=0.1)
    plazo_anios = st.sidebar.slider("Plazo del crédito (años)", 5, 30, 20, step=1)
else:
    # valores dummy para reutilizar funciones
    gastos_operitivos_dummy = 18.0
    ltv_pct = 70
    tasa_interes_anual = 10.5
    plazo_anios = 20

# =========================
# 5. MAPAS DE ZONA / SUPUESTOS
# =========================

riesgo_por_zona = {
    "Zibatá": 1,
    "Cumbres del Lago": 2,
    "Juriquilla": 2,
    "Jurica": 2,
    "El Refugio": 3,
    "Corregidora": 3,
    "Mileno III": 3,
    "Centro": 4
}
riesgo_zona = riesgo_por_zona.get(zona, 3)

yield_objetivo_zona = {
    "Zibatá": 7.5,
    "Cumbres del Lago": 7.2,
    "Juriquilla": 7.0,
    "Jurica": 6.8,
    "El Refugio": 7.3,
    "Corregidora": 7.8,
    "Mileno III": 7.9,
    "Centro": 8.2
}

apreciacion_zona = {
    "Zibatá": 4.0,
    "Cumbres del Lago": 3.8,
    "Juriquilla": 3.5,
    "Jurica": 3.2,
    "El Refugio": 3.0,
    "Corregidora": 2.8,
    "Mileno III": 2.5,
    "Centro": 2.2
}

yield_target = yield_objetivo_zona.get(zona, 7.0)
apreciacion_estimada = apreciacion_zona.get(zona, 3.0)

# =========================
# 6. PREDICCIONES DE LOS MODELOS
# =========================

# Para modelo de RENTAS (usa precio)
input_rent = pd.DataFrame([{
    "m2": m2,
    "recamaras": recamaras,
    "banos": banos,
    "estacionamientos": estacionamientos,
    "precio_venta_mxn": precio_venta_mxn,
    "antiguedad_anios": antiguedad_anios,
    "vacancia_pct": vacancia_pct,
    "riesgo_zona": riesgo_zona
}])

predicted_rent = model_rent.predict(input_rent)[0]

ajuste_zona = {
    "Zibatá": 1.10,
    "Cumbres del Lago": 1.08,
    "Juriquilla": 1.05,
    "Jurica": 1.03,
    "El Refugio": 1.00,
    "Corregidora": 0.97,
    "Mileno III": 0.95,
    "Centro": 0.98
}
factor_zona = ajuste_zona.get(zona, 1.0)
predicted_rent_ajustada = predicted_rent * factor_zona

# Para modelo de PRECIO (NO usa precio como input)
input_price = pd.DataFrame([{
    "m2": m2,
    "recamaras": recamaras,
    "banos": banos,
    "estacionamientos": estacionamientos,
    "antiguedad_anios": antiguedad_anios,
    "vacancia_pct": vacancia_pct,
    "riesgo_zona": riesgo_zona
}])

predicted_price = model_price.predict(input_price)[0]

# =========================
# 7. FUNCIONES AUXILIARES
# =========================

def pago_mensual(loan, annual_rate, years):
    if loan <= 0 or years <= 0:
        return 0.0
    r = annual_rate / 100.0 / 12.0
    n = years * 12
    if r == 0:
        return loan / n
    return loan * r / (1 - (1 + r) ** -n)

def calcular_score(rentabilidad, riesgo, vacancia, coc, dscr, objetivo):
    score = 0
    if rentabilidad >= objetivo + 1:
        score += 4
    elif rentabilidad >= objetivo - 0.3:
        score += 3
    elif rentabilidad >= objetivo - 1:
        score += 2
    else:
        score += 1
    if riesgo <= 2:
        score += 3
    elif riesgo == 3:
        score += 2
    else:
        score += 1
    if vacancia <= 4:
        score += 3
    elif vacancia <= 7:
        score += 2
    else:
        score += 1
    if coc >= 10:
        score += 3
    elif coc >= 7:
        score += 2
    elif coc >= 5:
        score += 1
    if dscr >= 1.5:
        score += 3
    elif dscr >= 1.2:
        score += 2
    elif dscr >= 1.0:
        score += 1
    return score

# =========================
# 8. LAYOUT SEGÚN MÓDULO
# =========================

if modulo == "Rentas":
    # ---------- Cálculos de rentas ----------
    ingreso_bruto_anual = predicted_rent_ajustada * 12 * (1 - vacancia_pct / 100.0)
    gastos_operativos_anuales = ingreso_bruto_anual * (gastos_operativos_pct / 100.0)
    noi = ingreso_bruto_anual - gastos_operativos_anuales

    cap_rate = (noi / precio_venta_mxn) * 100 if precio_venta_mxn > 0 else 0

    monto_prestamo = precio_venta_mxn * (ltv_pct / 100.0)
    equity = precio_venta_mxn - monto_prestamo

    pago_mensual_deuda = pago_mensual(monto_prestamo, tasa_interes_anual, plazo_anios)
    pago_anual_deuda = pago_mensual_deuda * 12

    ingreso_neto_despues_de_deuda = noi - pago_anual_deuda
    cash_on_cash = (ingreso_neto_despues_de_deuda / equity) * 100 if equity > 0 else 0

    dscr = noi / pago_anual_deuda if pago_anual_deuda > 0 else float("inf")

    score = calcular_score(cap_rate, riesgo_zona, vacancia_pct, cash_on_cash, dscr, yield_target)

    if score >= 13:
        etiqueta = "🟢 Excelente oportunidad"
    elif score >= 10:
        etiqueta = "🟡 Buena oportunidad"
    elif score >= 7:
        etiqueta = "🟠 Oportunidad ajustada"
    else:
        etiqueta = "🔴 Oportunidad débil"

    # ---------- Sensibilidad LTV ----------
    ltv_range = [50, 60, 70, 80, 90]
    sens_rows = []
    for l in ltv_range:
        mp = precio_venta_mxn * (l / 100.0)
        eq = precio_venta_mxn - mp
        pm_mes = pago_mensual(mp, tasa_interes_anual, plazo_anios)
        pa_anual = pm_mes * 12
        ds = noi / pa_anual if pa_anual > 0 else float("inf")
        coc_sens = (noi - pa_anual) / eq * 100 if eq > 0 else 0
        sens_rows.append([l, round(ds, 2), round(coc_sens, 2)])

    sens_df = pd.DataFrame(sens_rows, columns=["LTV (%)", "DSCR", "Cash-on-cash (%)"])

    # ---------- TABS MÓDULO RENTAS ----------
    tab_resumen, tab_analisis, tab_sens, tab_datos = st.tabs(
        ["📊 Resumen ejecutivo (rentas)", "📉 Análisis financiero", "📈 Sensibilidades", "📁 Base de datos"]
    )

    with tab_resumen:
        st.subheader("📊 Resumen ejecutivo del deal (Rentas)")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Renta mensual estimada", f"${predicted_rent_ajustada:,.0f} MXN")
            st.metric("Ingreso bruto anual", f"${ingreso_bruto_anual:,.0f} MXN")

        with col2:
            st.metric("Cap rate (NOI / Precio)", f"{cap_rate:,.2f} %")
            st.metric("Yield objetivo en zona", f"{yield_target:,.2f} %")

        with col3:
            st.metric("Cash-on-cash (aprox.)", f"{cash_on_cash:,.2f} %")
            st.metric("DSCR (NOI / Deuda)", f"{dscr:,.2f}x")

        st.markdown("---")

        st.markdown(f"### {etiqueta}")
        st.progress(min(score / 16, 1.0))

        comentario = []

        if cap_rate >= yield_target + 1:
            comentario.append("- La **rentabilidad bruta** está claramente por encima del objetivo para esta zona.")
        elif cap_rate >= yield_target - 0.3:
            comentario.append("- La **rentabilidad bruta** está alineada con el objetivo de la zona.")
        else:
            comentario.append("- La **rentabilidad bruta** queda por debajo del objetivo de la zona.")

        if cash_on_cash >= 10:
            comentario.append("- El **cash-on-cash** es atractivo para un inversor patrimonial apalancado.")
        elif cash_on_cash >= 7:
            comentario.append("- El **cash-on-cash** es razonable, pero podría mejorarse ajustando precio o LTV.")
        else:
            comentario.append("- El **cash-on-cash** es moderado, más cercano a un perfil defensivo que agresivo.")

        if dscr >= 1.5:
            comentario.append("- El **DSCR** ofrece un colchón cómodo frente al servicio de la deuda.")
        elif dscr >= 1.2:
            comentario.append("- El **DSCR** es aceptable, aunque sensible a desviaciones del NOI.")
        else:
            comentario.append("- El **DSCR** es ajustado; el deal es más sensible a cambios en renta o tipos.")

        st.markdown("\n".join(comentario))

        st.info(
            f"Suposición de apreciación anual implícita en **{zona}**: ~{apreciacion_estimada:.1f}%."
        )

    with tab_analisis:
        st.subheader("📉 Detalle del análisis financiero")

        col_izq, col_der = st.columns(2)

        with col_izq:
            st.markdown("#### Flujo operativo")
            st.write(f"- Renta mensual estimada: **${predicted_rent_ajustada:,.0f} MXN**")
            st.write(f"- Ingreso bruto anual (ajustado por vacancia): **${ingreso_bruto_anual:,.0f} MXN**")
            st.write(f"- Gastos operativos anuales ({gastos_operativos_pct:.1f}%): **${gastos_operativos_anuales:,.0f} MXN**")
            st.write(f"- NOI estimado: **${noi:,.0f} MXN**")

            st.markdown("#### Estructura de capital")
            st.write(f"- Precio de compra: **${precio_venta_mxn:,.0f} MXN**")
            st.write(f"- Monto del préstamo ({ltv_pct}% LTV): **${monto_prestamo:,.0f} MXN**")
            st.write(f"- Equity invertido: **${equity:,.0f} MXN**")

        with col_der:
            st.markdown("#### Servicio de la deuda")
            st.write(f"- Tasa de interés anual: **{tasa_interes_anual:.2f}%**")
            st.write(f"- Plazo: **{plazo_anios} años**")
            st.write(f"- Pago mensual estimado de deuda: **${pago_mensual_deuda:,.0f} MXN**")
            st.write(f"- Pago anual de deuda: **${pago_anual_deuda:,.0f} MXN**")

            st.markdown("#### Métricas clave")
            st.write(f"- Cap rate: **{cap_rate:,.2f}%**")
            st.write(f"- Cash-on-cash: **{cash_on_cash:,.2f}%**")
            st.write(f"- DSCR: **{dscr:,.2f}x**")

            if dscr < 1.1:
                st.error("El DSCR está por debajo de 1.1x: el flujo es frágil para una política de riesgo conservadora.")
            elif dscr < 1.3:
                st.warning("DSCR en zona intermedia (1.1x - 1.3x): financiable, pero con condiciones más estrictas.")
            else:
                st.success("DSCR sano: el deal sería atractivo para la mayoría de las entidades financieras.")

        st.markdown("---")

        comp_df = pd.DataFrame(
            {
                "Métrica": ["Yield objetivo en zona", "Cap rate estimado del deal"],
                "Valor (%)": [yield_target, cap_rate]
            }
        )
        st.markdown("#### Comparación con benchmark de la zona")
        st.table(comp_df.style.format({"Valor (%)": "{:.2f}"}))

    with tab_sens:
        st.subheader("📈 Sensibilidad LTV vs DSCR y Cash-on-cash")

        st.write(
            "Esta tabla muestra cómo se comportan el **DSCR** y el **cash-on-cash** al variar el nivel de apalancamiento (LTV)."
        )
        st.dataframe(sens_df)

        st.markdown("##### DSCR por nivel de LTV")
        st.bar_chart(sens_df.set_index("LTV (%)")["DSCR"])

        st.markdown("##### Cash-on-cash por nivel de LTV")
        st.bar_chart(sens_df.set_index("LTV (%)")["Cash-on-cash (%)"])

    with tab_datos:
        st.subheader("📁 Base interna de mercado residencial – Querétaro")
        st.write(
            """
            Vista tabular de la base utilizada para entrenar el modelo de estimación de rentas
            y calibrar supuestos de vacancia y riesgo por zona.
            """
        )
        st.dataframe(df)

# =========================
# MÓDULO PRECIO DE VENTA
# =========================

else:
    precio_recomendado = predicted_price
    precio_min = precio_recomendado * 0.95
    precio_max = precio_recomendado * 1.05

    ingreso_bruto_anual = predicted_rent_ajustada * 12 * (1 - vacancia_pct / 100.0)
    yield_bruto_estimado = (ingreso_bruto_anual / precio_recomendado) * 100 if precio_recomendado > 0 else 0

    delta_absoluto = precio_venta_mxn - precio_recomendado
    delta_pct = (delta_absoluto / precio_recomendado) * 100 if precio_recomendado > 0 else 0

    tab_resumen_v, tab_detalle_v, tab_datos_v = st.tabs(
        ["🏷️ Precio recomendado", "📉 Detalle de valoración", "📁 Base de datos"]
    )

    with tab_resumen_v:
        st.subheader("🏷️ Módulo Precio de venta – Vivienda nueva / reciente")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Precio recomendado IA", f"${precio_recomendado:,.0f} MXN")
            st.metric("Renta mensual estimada", f"${predicted_rent_ajustada:,.0f} MXN")

        with col2:
            st.metric("Rango sugerido mínimo", f"${precio_min:,.0f} MXN")
            st.metric("Rango sugerido máximo", f"${precio_max:,.0f} MXN")

        with col3:
            st.metric("Yield bruto estimado", f"{yield_bruto_estimado:,.2f} %")
            st.metric("Tu precio objetivo", f"${precio_venta_mxn:,.0f} MXN")

        st.markdown("---")

        texto_delta = "por encima" if delta_absoluto > 0 else "por debajo"
        st.markdown(
            f"""
            - Tu precio objetivo está **{texto_delta}** del precio recomendado por el modelo en aproximadamente **{abs(delta_pct):.2f}%**.
            - El rango de negociación razonable se sitúa entre **${precio_min:,.0f}** y **${precio_max:,.0f}** MXN.
            """
        )

        st.info(
            f"Suposición de apreciación anual en **{zona}**: ~{apreciacion_estimada:.1f}% (referencia estratégica, no input directo del modelo)."
        )

    with tab_detalle_v:
        st.subheader("📉 Detalle de la valoración de venta")

        col_izq, col_der = st.columns(2)

        with col_izq:
            st.markdown("#### Parámetros físicos")
            st.write(f"- Zona: **{zona}**")
            st.write(f"- Superficie: **{m2:.1f} m²**")
            st.write(f"- Recámaras: **{recamaras}**")
            st.write(f"- Baños: **{banos}**")
            st.write(f"- Estacionamientos: **{estacionamientos}**")
            st.write(f"- Antigüedad: **{antiguedad_anios} años**")

        with col_der:
            st.markdown("#### Indicadores de mercado")
            st.write(f"- Precio recomendado IA: **${precio_recomendado:,.0f} MXN**")
            st.write(f"- Renta estimada: **${predicted_rent_ajustada:,.0f} MXN/mes**")
            st.write(f"- Ingreso bruto anual ajustado por vacancia: **${ingreso_bruto_anual:,.0f} MXN**")
            st.write(f"- Yield bruto estimado sobre precio recomendado: **{yield_bruto_estimado:,.2f}%**")

            st.markdown("#### Comparación con tu precio objetivo")
            st.write(f"- Tu precio objetivo: **${precio_venta_mxn:,.0f} MXN**")
            st.write(f"- Diferencia absoluta: **${delta_absoluto:,.0f} MXN**")
            st.write(f"- Diferencia relativa: **{delta_pct:,.2f}%**")

    with tab_datos_v:
        st.subheader("📁 Base interna de mercado residencial – Querétaro")
        st.write(
            """
            Esta base soporta tanto la estimación de renta como la estimación de precio de venta
            para inmuebles residenciales en Querétaro.
            """
        )
        st.dataframe(df)
