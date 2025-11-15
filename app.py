
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# =========================
# 1. BASE DE DATOS DE MERCADO – QUERÉTARO
# =========================

@st.cache_data
def cargar_base():
    return pd.read_excel("base_mercado_qro.xlsx")

df = cargar_base()

# =========================
# 2. ENTRENAR MODELO DE ML
# =========================

feature_cols = ["m2", "recamaras", "banos", "estacionamientos",
                "precio_venta_mxn", "antiguedad_anios", "vacancia_pct", "riesgo_zona"]
X = df[feature_cols]
y = df["renta_mensual_mxn"]

model = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    max_depth=10
)
model.fit(X, y)

# =========================
# 3. CONFIGURACIÓN DE PÁGINA
# =========================

st.set_page_config(
    page_title="AI Deal Screener – Querétaro",
    layout="wide"
)

st.title("🏙️ AI Deal Screener – Querétaro")
st.caption("Herramienta interna para análisis rápido de oportunidades de inversión residencial con modelos de IA / ML")

st.markdown(
    '''
    Panel diseñado como un **copiloto de análisis** para inversiones residenciales en Querétaro.

    - Modelo de **Random Forest** entrenado con una base interna de mercado residencial.
    - Estima **renta mensual**, **rentabilidad bruta**, **NOI, cap rate, cash-on-cash y DSCR**.
    - Integra supuestos de **vacancia, gastos operativos y deuda** para aproximar el análisis real de un deal.
    '''
)

# =========================
# 4. SIDEBAR: INPUTS DEL DEAL
# =========================

st.sidebar.header("🎯 Parámetros del deal")

zona = st.sidebar.selectbox(
    "Zona",
    sorted(df["zona"].unique())
)

m2 = st.sidebar.slider("Metros cuadrados construidos", 40, 260, 110, step=5)
recamaras = st.sidebar.slider("Recámaras", 1, 5, 3)
banos = st.sidebar.slider("Baños", 1, 4, 2)
estacionamientos = st.sidebar.slider("Estacionamientos", 0, 4, 2)

precio_venta_mxn = st.sidebar.number_input(
    "Precio de compra (MXN)",
    min_value=800000,
    max_value=20000000,
    value=3800000,
    step=50000
)

antiguedad_anios = st.sidebar.slider("Antigüedad (años)", 0, 40, 8)
vacancia_pct = st.sidebar.slider("Vacancia estimada (%)", 0.0, 20.0, 5.0, step=0.5)
gastos_operativos_pct = st.sidebar.slider("Gastos operativos (% del ingreso bruto)", 5.0, 40.0, 18.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Supuestos de financiamiento")

ltv_pct = st.sidebar.slider("LTV (porcentaje financiado)", 0, 90, 70, step=5)
tasa_interes_anual = st.sidebar.slider("Tasa de interés anual (%)", 5.0, 16.0, 10.5, step=0.1)
plazo_anios = st.sidebar.slider("Plazo del crédito (años)", 5, 30, 20, step=1)

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
# 6. PREDICCIÓN DEL MODELO
# =========================

input_data = pd.DataFrame([{
    "m2": m2,
    "recamaras": recamaras,
    "banos": banos,
    "estacionamientos": estacionamientos,
    "precio_venta_mxn": precio_venta_mxn,
    "antiguedad_anios": antiguedad_anios,
    "vacancia_pct": vacancia_pct,
    "riesgo_zona": riesgo_zona
}])

predicted_rent = model.predict(input_data)[0]

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

# =========================
# 7. CÁLCULOS FINANCIEROS
# =========================

ingreso_bruto_anual = predicted_rent_ajustada * 12 * (1 - vacancia_pct / 100.0)
gastos_operativos_anuales = ingreso_bruto_anual * (gastos_operativos_pct / 100.0)
noi = ingreso_bruto_anual - gastos_operativos_anuales

cap_rate = (noi / precio_venta_mxn) * 100 if precio_venta_mxn > 0 else 0

monto_prestamo = precio_venta_mxn * (ltv_pct / 100.0)
equity = precio_venta_mxn - monto_prestamo

def pago_mensual(loan, annual_rate, years):
    if loan <= 0 or years <= 0:
        return 0.0
    r = annual_rate / 100.0 / 12.0
    n = years * 12
    if r == 0:
        return loan / n
    return loan * r / (1 - (1 + r) ** -n)

pago_mensual_deuda = pago_mensual(monto_prestamo, tasa_interes_anual, plazo_anios)
pago_anual_deuda = pago_mensual_deuda * 12

ingreso_neto_despues_de_deuda = noi - pago_anual_deuda
cash_on_cash = (ingreso_neto_despues_de_deuda / equity) * 100 if equity > 0 else 0

dscr = noi / pago_anual_deuda if pago_anual_deuda > 0 else float("inf")

# Score global del deal
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

score = calcular_score(cap_rate, riesgo_zona, vacancia_pct, cash_on_cash, dscr, yield_target)

if score >= 13:
    etiqueta = "🟢 Excelente oportunidad"
elif score >= 10:
    etiqueta = "🟡 Buena oportunidad"
elif score >= 7:
    etiqueta = "🟠 Oportunidad ajustada"
else:
    etiqueta = "🔴 Oportunidad débil"

# =========================
# 8. SENSIBILIDAD SIMPLE LTV vs DSCR / CASH-ON-CASH
# =========================

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

# =========================
# 9. LAYOUT PRINCIPAL: TABS
# =========================

tab_resumen, tab_analisis, tab_sens, tab_datos = st.tabs(
    ["📊 Resumen ejecutivo", "📉 Análisis financiero", "📈 Sensibilidades", "📁 Base de datos de mercado"]
)

with tab_resumen:
    st.subheader("📊 Resumen ejecutivo del deal")

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
        '''
        Vista tabular de la base utilizada para entrenar el modelo de estimación de rentas
        y calibrar supuestos de vacancia y riesgo por zona.
        '''
    )
    st.dataframe(df)
