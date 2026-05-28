"""
================================================================================
  ANALIZADOR DE PORTAFOLIOS & VALORACIÓN DE ACTIVOS
  Autor: Diego CR
  Descripción: Aplicación Streamlit para análisis de portafolios, optimización
               y valoración técnica, estadística y fundamental de activos.
================================================================================
"""

# ── Librerías estándar ────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import datetime
import requests

# ── Librerías de datos y cálculo ─────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import statsmodels.api as sm

# ── Descarga de datos financieros ────────────────────────────────────────────
import yfinance as yf

# ── Visualización ────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Interfaz Streamlit ───────────────────────────────────────────────────────
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analizador de Portafolios",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <div style='text-align:right; color:#888; font-size:0.78rem; padding-bottom:4px;'>
        Desarrollado por <strong>Diego CR</strong> · Análisis de Portafolios & Valoración
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
TASA_LIBRE_RIESGO = 0.0457
DIAS_ANIO = 252
PERIODOS_LABEL = {
    "1 Año": 1, "3 Años": 3, "5 Años": 5,
    "7 Años": 7, "10 Años": 10, "15 Años": 15, "20 Años": 20,
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR + SESSION_STATE
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Configuración")
    st.caption("Ingresa los tickers y parámetros del análisis")

    tickers_input = st.text_area(
        "Tickers del portafolio (separados por coma)",
        value="XLK, QQQ, IWM, EFA, EEM",
        help="Ejemplo: AAPL, MSFT, AMZN, GOOGL",
    )
    benchmark_input = st.text_input(
        "Ticker del Benchmark",
        value="^GSPC",
        help="Índice de referencia. Ejemplo: ^GSPC para S&P 500",
    )
    periodo = st.selectbox(
        "Período de análisis",
        options=list(PERIODOS_LABEL.keys()),
        index=2,
        help="Ventana histórica de precios a descargar",
    )
    rf = st.number_input(
        "Tasa libre de riesgo (anual, decimal)",
        value=TASA_LIBRE_RIESGO,
        step=0.001,
        format="%.4f",
    )

    btn_analizar = st.button("🚀 Analizar Portafolio", type="primary", use_container_width=True)
    st.divider()
    st.caption("**Autor:** Diego CR")

if "analizar" not in st.session_state:
    st.session_state.analizar = False

if btn_analizar:
    st.session_state.analizar = True

if not st.session_state.analizar:
    st.markdown("## 📈 Analizador de Portafolios & Valoración")
    st.markdown("""
**Bienvenido.** Esta herramienta permite:

- 📊 Análisis de portafolio  
- ⚙️ Optimización  
- 🎲 Simulación Monte Carlo  
- 📉 Valoración técnica  
- 📐 Valoración estadística  
- 🏦 Valoración fundamental  
- ✅ Recomendación final  

👈 Ingresa los tickers y haz clic en **Analizar Portafolio**.
""")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: TRM Colombia
# ─────────────────────────────────────────────────────────────────────────────
def obtener_trm_colombia():
    try:
        url = "https://www.datos.gov.co/resource/32sa-8pi3.json?$limit=1&$order=vigenciadesde DESC"
        r = requests.get(url, timeout=5)
        data = r.json()
        return float(data[0]["valor"])
    except:
        return np.nan

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Clasificación de tickers
# ─────────────────────────────────────────────────────────────────────────────
def clasificar_ticker(ticker):
    try:
        info = yf.Ticker(ticker).info
    except:
        info = {}

    qt = str(info.get("quoteType", "")).upper()
    fund = info.get("fundFamily")
    region = info.get("region", "N/D")
    name = info.get("longName", ticker)

    if qt == "ETF" or fund:
        tipo = "ETF"
        emisor = fund or "N/D"
    elif qt == "EQUITY":
        tipo = "Acción"
        emisor = name
    else:
        tipo = qt or "Otro"
        emisor = name

    return {"Ticker": ticker, "Tipo": tipo, "Región": region, "Emisor": emisor}

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Tipo de activo
# ─────────────────────────────────────────────────────────────────────────────
def tipo_activo_desde_info(info):
    qt = str(info.get("quoteType", "")).upper()
    if qt == "ETF" or info.get("fundFamily"):
        return "ETF"
    elif qt == "EQUITY":
        return "Acción"
    return qt or "Otro"

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Valoración fundamental ETF
# ─────────────────────────────────────────────────────────────────────────────
def valoracion_fundamental_etf(ticker):
    try:
        info = yf.Ticker(ticker).info
    except:
        info = {}

    return {
        "nombre": info.get("longName", ticker),
        "emisor": info.get("fundFamily", "N/D"),
        "precio": info.get("currentPrice", np.nan),
        "AUM": info.get("totalAssets", np.nan),
        "Expense Ratio": info.get("expenseRatio", np.nan),
        "Beta": info.get("beta", np.nan),
        "Categoria": info.get("category", "N/D"),
        "Region": info.get("region", "N/D"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Wrapper ETF/Acción
# ─────────────────────────────────────────────────────────────────────────────
def valoracion_fundamental_general(ticker):
    try:
        info = yf.Ticker(ticker).info
    except:
        info = {}

    tipo = tipo_activo_desde_info(info)

    if tipo == "ETF":
        return valoracion_fundamental_etf(ticker)
    return valoracion_fundamental_basica(ticker)
# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Valoración fundamental (Acciones)
# ─────────────────────────────────────────────────────────────────────────────
def valoracion_fundamental_basica(ticker):
    try:
        info = yf.Ticker(ticker).info
    except:
        info = {}

    return {
        "nombre": info.get("longName", ticker),
        "sector": info.get("sector", "N/D"),
        "precio": info.get("currentPrice", np.nan),
        "P/E": info.get("trailingPE", np.nan),
        "P/B": info.get("priceToBook", np.nan),
        "EV/EBITDA": info.get("enterpriseToEbitda", np.nan),
        "ROE": info.get("returnOnEquity", np.nan),
        "Beta": info.get("beta", np.nan),
    }

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Rendimientos logarítmicos
# ─────────────────────────────────────────────────────────────────────────────
def calcular_rendimientos(precios: pd.DataFrame) -> pd.DataFrame:
    return np.log(precios / precios.shift(1)).dropna()

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Base 100
# ─────────────────────────────────────────────────────────────────────────────
def base_100(precios: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    if precios.empty or len(precios) == 0:
        return precios
    primer_valor = precios.iloc[0].replace(0, np.nan)
    return (precios / primer_valor) * base

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Métricas anualizadas
# ─────────────────────────────────────────────────────────────────────────────
def metricas_anuales(rendimientos: pd.DataFrame, rf_diaria: float) -> pd.DataFrame:
    media   = rendimientos.mean()
    vol_d   = rendimientos.std()
    rend_ea = np.exp(media * DIAS_ANIO) - 1
    vol_ea  = vol_d * np.sqrt(DIAS_ANIO)
    sharpe  = (rend_ea - rf_diaria * DIAS_ANIO) / vol_ea.replace(0, np.nan)

    filas = {}
    for col in rendimientos.columns:
        filas[col] = {
            "Rendimiento EA": "{:.2%}".format(rend_ea[col]),
            "Volatilidad EA": "{:.2%}".format(vol_ea[col]),
            "Sharpe Ratio":   "{:.4f}".format(sharpe[col]),
        }
    return pd.DataFrame(filas).T

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Métricas de riesgo (Beta, VaR, CVaR)
# ─────────────────────────────────────────────────────────────────────────────
def metricas_riesgo(rendimientos: pd.DataFrame, benchmark: str,
                    nivel: float = 0.05) -> pd.DataFrame:
    if benchmark not in rendimientos.columns:
        return pd.DataFrame()

    activos = [c for c in rendimientos.columns if c != benchmark]
    r_bmk   = rendimientos[benchmark].dropna().values

    resultados = {}
    for ticker in activos:
        r = rendimientos[ticker].dropna().values

        n = min(len(r), len(r_bmk))
        if n < 5:
            resultados[ticker] = {"Beta": "N/D", "VaR 95% (1d)": "N/D", "CVaR 95% (1d)": "N/D"}
            continue

        r_a = r[-n:]
        r_b = r_bmk[-n:]

        cov_mat = np.cov(r_a, r_b)
        var_b   = cov_mat[1, 1]
        beta    = cov_mat[0, 1] / var_b if var_b > 0 else np.nan

        var_hist = float(np.nanpercentile(r_a, nivel * 100))

        cola = r_a[r_a <= var_hist]
        cvar = float(np.mean(cola)) if len(cola) > 0 else var_hist

        resultados[ticker] = {
            "Beta":           f"{beta:.4f}" if not np.isnan(beta) else "N/D",
            "VaR 95% (1d)":   f"{var_hist*100:.2f}%",
            "CVaR 95% (1d)":  f"{cvar*100:.2f}%",
        }

    return pd.DataFrame(resultados).T

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Descargar precios (robusta)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def descargar_precios(tickers: list, benchmark: str, anios: int) -> pd.DataFrame:
    fecha_fin    = datetime.date.today()
    fecha_inicio = fecha_fin - datetime.timedelta(days=int(anios * 365.25))

    todos = list(dict.fromkeys([t.upper() for t in tickers] + [benchmark.upper()]))

    def _extraer_close(raw_df: pd.DataFrame, simbolo: str) -> pd.Series:
        if isinstance(raw_df.columns, pd.MultiIndex):
            lvl0 = raw_df.columns.get_level_values(0).unique().tolist()
            lvl1 = raw_df.columns.get_level_values(1).unique().tolist()
            if "Close" in lvl0:
                close = raw_df["Close"]
                if isinstance(close, pd.DataFrame):
                    if simbolo in close.columns:
                        return close[simbolo]
                    for c in close.columns:
                        if str(c).upper() == simbolo.upper():
                            return close[c]
                return close.squeeze()
            elif "Close" in lvl1:
                if simbolo in lvl0:
                    return raw_df[simbolo]["Close"]
                for t in lvl0:
                    if str(t).upper() == simbolo.upper():
                        return raw_df[t]["Close"]
        else:
            if "Close" in raw_df.columns:
                return raw_df["Close"]
        return pd.Series(dtype=float)

    try:
        raw = yf.download(
            todos,
            start=str(fecha_inicio),
            end=str(fecha_fin),
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        series_dict = {}
        for t in todos:
            s = _extraer_close(raw, t)
            s = pd.to_numeric(s, errors="coerce")
            if not s.dropna().empty:
                series_dict[t] = s
    except Exception:
        series_dict = {}

    if len(series_dict) < 2:
        series_dict = {}
        for t in todos:
            try:
                tmp = yf.download(t, start=str(fecha_inicio), end=str(fecha_fin),
                                  auto_adjust=True, progress=False)
                s = _extraer_close(tmp, t)
                s = pd.to_numeric(s, errors="coerce").squeeze()
                if not s.dropna().empty:
                    series_dict[t] = s
            except Exception:
                pass

    if not series_dict:
        return pd.DataFrame()

    precios = pd.DataFrame(series_dict)
    precios.index.name = "Date"

    if isinstance(precios.columns, pd.MultiIndex):
        precios.columns = [
            str(c[0]).upper() if isinstance(c, tuple) else str(c).upper()
            for c in precios.columns
        ]

    precios = precios.apply(pd.to_numeric, errors="coerce").dropna()

    return precios
# ═════════════════════════════════════════════════════════════════════════════
#  PROCESAMIENTO PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════
tickers_raw = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
benchmark   = benchmark_input.strip().upper()
anios       = PERIODOS_LABEL[periodo]
rf_diaria   = rf / DIAS_ANIO

if len(tickers_raw) < 2:
    st.error("❌ Por favor ingresa al menos 2 tickers.")
    st.stop()

tickers = [t for t in tickers_raw if t != benchmark]

with st.spinner("⏳ Descargando precios desde Yahoo Finance..."):
    try:
        precios_full = descargar_precios(tickers, benchmark, anios)
    except Exception as e:
        st.error(f"❌ Error al descargar datos: {e}")
        st.stop()

if precios_full.empty:
    st.error("❌ No se pudieron descargar datos. Verifica los símbolos e intenta de nuevo.")
    st.stop()

tickers_ok   = [t for t in tickers if t in precios_full.columns]
benchmark_ok = benchmark in precios_full.columns

if not tickers_ok:
    st.error("❌ Ningún ticker del portafolio devolvió datos.")
    st.stop()

precios_port      = precios_full[tickers_ok]
rendimientos_full = calcular_rendimientos(precios_full)
rendimientos_port = rendimientos_full[tickers_ok]

st.success(
    f"✅ Datos descargados: **{len(precios_full)} días** | "
    f"Tickers: {', '.join(tickers_ok)}"
    + (f" + benchmark {benchmark}" if benchmark_ok else "")
)

# ═════════════════════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🌍 Macro & Resumen",
    "📊 Análisis de Portafolio",
    "⚙️ Optimización",
    "🎲 Monte Carlo",
    "🔍 Valoración Ticker",
])

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 0: MACRO & RESUMEN
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("🌍 Indicadores Macroeconómicos & Resumen de Tickers")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💱 TRM COP/USD")
        trm = obtener_trm_colombia()
        if np.isnan(trm):
            st.warning("No se pudo obtener la TRM actual.")
        else:
            st.metric("TRM Hoy (COP/USD)", f"{trm:,.2f}")
        st.caption("Fuente: datos.gov.co")

    with col2:
        st.subheader("📋 Clasificación de Tickers")
        resumen = [clasificar_ticker(t) for t in tickers_raw]
        df_resumen = pd.DataFrame(resumen)
        st.dataframe(df_resumen, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1: ANÁLISIS DE PORTAFOLIO
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("📊 Análisis del Portafolio")

    st.subheader("Precios normalizados (Base 100)")
    precios_b100 = base_100(precios_full, 100)
    df_b100 = precios_b100.reset_index()
    fig_b100 = px.line(
        df_b100, x="Date", y=precios_b100.columns.tolist(),
        title="Evolución de activos en base 100",
        labels={"value": "Valor (base 100)", "Date": "Fecha"},
        template="plotly_dark",
    )
    st.plotly_chart(fig_b100, use_container_width=True)

    st.subheader("Rendimientos acumulados")
    rend_acum = np.exp(rendimientos_full.cumsum()) - 1
    df_rend   = rend_acum.reset_index()
    fig_rend  = px.line(
        df_rend, x="Date", y=rend_acum.columns.tolist(),
        title="Rendimiento acumulado",
        labels={"value": "Rendimiento acumulado", "Date": "Fecha"},
        template="plotly_dark",
    )
    fig_rend.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_rend, use_container_width=True)

    st.subheader("Métricas anualizadas")
    met_df = metricas_anuales(rendimientos_full, rf_diaria)
    st.dataframe(met_df, use_container_width=True)

    if benchmark_ok:
        st.subheader(f"Métricas de riesgo vs. {benchmark}")
        tabla_riesgo = metricas_riesgo(rendimientos_full, benchmark)
        st.dataframe(tabla_riesgo, use_container_width=True)

    st.subheader("Matriz de correlación")
    corr = rendimientos_port.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlación entre activos",
        template="plotly_dark",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    if benchmark_ok:
        st.subheader(f"Portafolio igual peso vs. {benchmark}")
        w_eq         = np.ones(len(tickers_ok)) / len(tickers_ok)
        rend_ew      = rendimientos_port @ w_eq
        acum_port    = np.exp(rend_ew.cumsum()) - 1
        acum_bmk     = np.exp(rendimientos_full[benchmark].cumsum()) - 1

        fig_vs = go.Figure()
        fig_vs.add_trace(go.Scatter(x=acum_port.index, y=acum_port,
                                    name="Portafolio (EW)", line=dict(color="cyan")))
        fig_vs.add_trace(go.Scatter(x=acum_bmk.index, y=acum_bmk,
                                    name=benchmark, line=dict(color="orange")))
        fig_vs.update_layout(title="Portafolio igual peso vs Benchmark",
                             yaxis_tickformat=".0%", template="plotly_dark")
        st.plotly_chart(fig_vs, use_container_width=True)
# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2: OPTIMIZACIÓN
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("⚙️ Optimización de Portafolio")
    st.info("Tres métodos: Markowitz (mín varianza), CAPM / Máx Sharpe, y Monte Carlo.")

    with st.spinner("Optimizando portafolios..."):
        mk_res = portafolio_markowitz(rendimientos_port, rf)
        ms_res = portafolio_maximo_sharpe(rendimientos_port, rf)
        mc_res = portafolio_montecarlo(rendimientos_port, rf)

    st.subheader("Resumen comparativo")
    st.dataframe(
        tabla_comparacion_portafolios(tickers_ok, mk_res, ms_res, mc_res),
        use_container_width=True
    )

    st.subheader("Distribución de pesos por método")
    df_pesos = pd.DataFrame({
        "Ticker":        tickers_ok,
        "Markowitz":     mk_res["pesos"] * 100,
        "CAPM / Sharpe": ms_res["pesos"] * 100,
        "Montecarlo":    mc_res["pesos"] * 100,
    }).melt(id_vars="Ticker", var_name="Método", value_name="Peso (%)")

    fig_pesos = px.bar(
        df_pesos, x="Ticker", y="Peso (%)", color="Método",
        barmode="group", title="Pesos óptimos (%)", template="plotly_dark"
    )
    st.plotly_chart(fig_pesos, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    for col, nombre, res in [
        (col1, "Markowitz (Mín Var.)", mk_res),
        (col2, "CAPM / Máx Sharpe",   ms_res),
        (col3, "Monte Carlo",         mc_res),
    ]:
        with col:
            st.markdown(f"**{nombre}**")
            st.write(f"Rendimiento EA: {res['rendimiento']*100:.2f}%")
            st.write(f"Volatilidad EA: {res['volatilidad']*100:.2f}%")
            st.write(f"Sharpe Ratio:   {res['sharpe']:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3: MONTE CARLO
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("🎲 Simulación Monte Carlo de Precios (1 año)")
    st.caption("1 000 trayectorias usando movimiento browniano geométrico (GBM).")

    ticker_mc = st.selectbox("Selecciona el ticker a simular", options=tickers_ok, key="mc_sel")

    with st.spinner("Simulando..."):
        r_tick   = rendimientos_port[ticker_mc]
        p_actual = float(precios_port[ticker_mc].iloc[-1])
        sim      = montecarlo_precio(r_tick, p_actual, n_sim=1000, horizonte=252)

    p5  = float(sim.iloc[-1].quantile(0.05))
    p50 = float(sim.iloc[-1].quantile(0.50))
    p95 = float(sim.iloc[-1].quantile(0.95))

    fig_mc = go.Figure()
    for col in sim.columns[:200]:
        fig_mc.add_trace(go.Scatter(
            y=sim[col], mode="lines",
            line=dict(width=0.5, color="rgba(100,180,255,0.15)"),
            showlegend=False,
        ))
    fig_mc.add_hline(y=p5,  line_dash="dot", line_color="red",
                     annotation_text=f"P5: ${p5:.2f}")
    fig_mc.add_hline(y=p50, line_dash="dot", line_color="yellow",
                     annotation_text=f"P50: ${p50:.2f}")
    fig_mc.add_hline(y=p95, line_dash="dot", line_color="green",
                     annotation_text=f"P95: ${p95:.2f}")
    fig_mc.update_layout(
        title=f"Monte Carlo: {ticker_mc} – 1 000 trayectorias a 252 días",
        yaxis_title="Precio simulado (USD)", xaxis_title="Días hábiles",
        template="plotly_dark",
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precio actual",   f"${p_actual:.2f}")
    c2.metric("P5  (bajista)",   f"${p5:.2f}",  f"{(p5/p_actual-1)*100:.1f}%")
    c3.metric("P50 (base)",      f"${p50:.2f}", f"{(p50/p_actual-1)*100:.1f}%")
    c4.metric("P95 (alcista)",   f"${p95:.2f}", f"{(p95/p_actual-1)*100:.1f}%")
    c5.metric("Media simulada",  f"${sim.iloc[-1].mean():.2f}")

    fig_hist = px.histogram(
        sim.iloc[-1], nbins=50,
        title=f"Distribución de precios finales – {ticker_mc}",
        labels={"value": "Precio final (USD)"},
        template="plotly_dark", color_discrete_sequence=["#4FC3F7"],
    )
    fig_hist.add_vline(x=p_actual, line_dash="dash", line_color="white",
                       annotation_text="Precio actual")
    st.plotly_chart(fig_hist, use_container_width=True)
# ═════════════════════════════════════════════════════════════════════════════
#  TAB 4: VALORACIÓN TICKER
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🔍 Valoración de Ticker Individual")

    ticker_val = st.selectbox("Selecciona el ticker a valorar", options=tickers_ok, key="val_sel")

    st.subheader("🎛️ Ponderación de métodos")
    st.caption("La suma debe ser 100%. Se normaliza automáticamente si no lo es.")
    c1, c2, c3 = st.columns(3)
    with c1:
        peso_tec  = st.slider("Análisis Técnico (%)",     0, 100, 33, step=1) / 100
    with c2:
        peso_est  = st.slider("Análisis Estadístico (%)", 0, 100, 33, step=1) / 100
    with c3:
        peso_fund = st.slider("Análisis Fundamental (%)", 0, 100, 34, step=1) / 100

    suma_pesos = peso_tec + peso_est + peso_fund
    if abs(suma_pesos - 1.0) > 0.01:
        st.warning(f"⚠️ Los pesos suman {suma_pesos*100:.1f}%. Se normalizarán.")

    # ─────────────────────────────────────────────────────────────────────────
    # ANÁLISIS TÉCNICO
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📉 1. Valoración Técnica")

    precios_t = precios_full[ticker_val]
    ind_df    = calcular_indicadores_tecnicos(precios_t)
    sen       = senal_tecnica(ind_df)

    fig_tec = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             row_heights=[0.5, 0.25, 0.25],
                             subplot_titles=[
                                 f"{ticker_val} – Precio y Medias Móviles",
                                 "RSI (14)", "MACD"])

    fig_tec.add_trace(go.Scatter(x=ind_df.index, y=ind_df["Precio"],
                                  name="Precio", line=dict(color="white", width=1)), row=1, col=1)
    for ma, color in [("MA5","cyan"),("MA10","yellow"),("MA20","orange"),("MA200","red")]:
        fig_tec.add_trace(go.Scatter(x=ind_df.index, y=ind_df[ma],
                                      name=ma, line=dict(color=color, width=1.2)), row=1, col=1)
    for nombre_f, val_f in sen["fibo_niveles"].items():
        fig_tec.add_hline(y=val_f, line_dash="dot",
                           line_color="rgba(255,215,0,0.35)",
                           annotation_text=nombre_f,
                           annotation_font_size=9, row=1, col=1)

    fig_tec.add_trace(go.Scatter(x=ind_df.index, y=ind_df["RSI"],
                                  name="RSI", line=dict(color="violet")), row=2, col=1)
    fig_tec.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
    fig_tec.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    colores_hist = ["green" if v >= 0 else "red" for v in ind_df["Hist"].fillna(0)]
    fig_tec.add_trace(go.Bar(x=ind_df.index, y=ind_df["Hist"],
                              name="Histograma", marker_color=colores_hist), row=3, col=1)
    fig_tec.add_trace(go.Scatter(x=ind_df.index, y=ind_df["MACD"],
                                  name="MACD", line=dict(color="blue")), row=3, col=1)
    fig_tec.add_trace(go.Scatter(x=ind_df.index, y=ind_df["Signal"],
                                  name="Signal", line=dict(color="orange")), row=3, col=1)
    fig_tec.update_layout(height=700, template="plotly_dark")
    st.plotly_chart(fig_tec, use_container_width=True)

    senales_df = pd.DataFrame([
        {"Indicador": k,
         "Señal": "🟢 ALCISTA" if v > 0 else ("🔴 BAJISTA" if v < 0 else "⚪ NEUTRAL"),
         "Puntuación": v}
        for k, v in sen["señales"].items()
    ])
    st.dataframe(senales_df, use_container_width=True)
    st.write(f"**Score técnico:** {sen['score_tecnico']:.4f}")
    score_tecnico = sen["score_tecnico"]

    # ─────────────────────────────────────────────────────────────────────────
    # ANÁLISIS ESTADÍSTICO
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📐 2. Valoración Estadística")
    score_estadistico = 0.0

    if benchmark_ok:
        anios_reg = st.slider("Horizonte de regresión (años)", 1, min(anios, 5), 1)
        est = valoracion_estadistica(precios_full[ticker_val],
                                     precios_full[benchmark], anios_reg)
        if est:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precio actual",      f"${est['precio_actual']:.2f}")
            c2.metric("PO Regresión",        f"${est['precio_obj_reg']:.2f}",
                       f"{est['potencial_reg']*100:+.2f}%")
            c3.metric("R²", f"{est['r2']:.4f}",
                       "Confiable" if est["confiable"] else "Baja conf.")
            c4.metric("Percentil actual",   f"{est['percentil_actual']*100:.1f}%")

            ln_t_all = np.log(precios_full[ticker_val])
            ln_b_all = np.log(precios_full[benchmark])
            x_lin = np.linspace(float(ln_b_all.min()), float(ln_b_all.max()), 100)
            y_lin = est["alfa"] + est["beta"] * x_lin

            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=ln_b_all, y=ln_t_all, mode="markers",
                                          name="Obs.",
                                          marker=dict(color="cyan", size=3, opacity=0.4)))
            fig_reg.add_trace(go.Scatter(x=x_lin, y=y_lin, mode="lines",
                                          name=f"Regresión (R²={est['r2']:.3f})",
                                          line=dict(color="orange")))
            fig_reg.update_layout(title=f"Regresión LN {ticker_val} vs LN {benchmark}",
                                   xaxis_title=f"ln({benchmark})",
                                   yaxis_title=f"ln({ticker_val})",
                                   template="plotly_dark")
            st.plotly_chart(fig_reg, use_container_width=True)

            perc_labels = [f"P{int(k*100)}" for k in est["percentiles"]]
            perc_vals   = list(est["percentiles"].values())
            fig_perc = go.Figure()
            fig_perc.add_trace(go.Bar(x=perc_labels, y=perc_vals,
                                       marker_color="steelblue"))
            fig_perc.add_hline(y=est["precio_actual"], line_color="red", line_dash="dash",
                                annotation_text=f"Precio actual: ${est['precio_actual']:.2f}")
            fig_perc.update_layout(title="Percentiles históricos de precio",
                                    template="plotly_dark")
            st.plotly_chart(fig_perc, use_container_width=True)

            score_reg  = float(np.clip(est["potencial_reg"], -1, 1))
            score_perc = float(np.clip(1 - 2 * est["percentil_actual"], -1, 1))
            score_estadistico = 0.6 * score_reg + 0.4 * score_perc
            st.write(f"**Score estadístico:** {score_estadistico:.4f}")
        else:
            st.warning("Datos insuficientes para la regresión.")
    else:
        st.warning("Benchmark no disponible para valoración estadística.")

    # ─────────────────────────────────────────────────────────────────────────
    # ANÁLISIS FUNDAMENTAL (ETF o Acción)
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("🏦 3. Valoración Fundamental")

    with st.spinner("Descargando datos fundamentales..."):
        fund = valoracion_fundamental_general(ticker_val)

    if "AUM" in fund:   # ETF
        st.markdown(f"**{fund['nombre']}** | Emisor: {fund['emisor']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${fund['precio']:.2f}" if not pd.isna(fund['precio']) else "N/D")
        c2.metric("AUM", f"${fund['AUM']:,}" if not pd.isna(fund['AUM']) else "N/D")
        c3.metric("Expense Ratio", f"{fund['Expense Ratio']*100:.2f}%" if not pd.isna(fund['Expense Ratio']) else "N/D")
        c4.metric("Beta", f"{fund['Beta']:.2f}" if not pd.isna(fund['Beta']) else "N/D")

        score_fundamental = 0.0
        if not pd.isna(fund["Expense Ratio"]):
            score_fundamental += (0.5 if fund["Expense Ratio"] < 0.003 else -0.5)

        st.info("ETF detectado: se omite DCF y múltiplos de empresa.")

    else:  # Acción
        st.markdown(f"**{fund['nombre']}** | Sector: {fund['sector']}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Precio", f"${fund['precio']:.2f}"    if not pd.isna(fund['precio'])    else "N/D")
        c2.metric("P/E",    f"{fund['P/E']:.1f}x"       if not pd.isna(fund['P/E'])       else "N/D")
        c3.metric("P/B",    f"{fund['P/B']:.2f}x"       if not pd.isna(fund['P/B'])       else "N/D")
        c4.metric("EV/EBITDA", f"{fund['EV/EBITDA']:.1f}x" if not pd.isna(fund['EV/EBITDA']) else "N/D")
        c5.metric("ROE",    f"{fund['ROE']*100:.1f}%"   if not pd.isna(fund['ROE'])       else "N/D")

        st.markdown("#### DCF – Flujo de Caja Libre Descontado")
        st.caption("Ingresa los datos del último reporte para calcular el valor intrínseco.")

        score_fundamental = 0.0

        with st.expander("📋 Ingresar datos financieros para DCF (opcional)"):
            c1, c2 = st.columns(2)
            with c1:
                uo         = st.number_input("Utilidad Operativa (EBIT) 12M",       value=0.0, format="%.0f")
                dda        = st.number_input("Depreciación y Amortización 12M",     value=0.0, format="%.0f")
                capex      = st.number_input("CAPEX 12M",                            value=0.0, format="%.0f")
                delta_ktno = st.number_input("Variación KTNO",                      value=0.0, format="%.0f")
                tasa_imp   = st.number_input("Tasa de impuestos (decimal)",          value=0.25, format="%.4f")
            with c2:
                rf_dcf   = st.number_input("Tasa libre de riesgo",      value=rf,   format="%.4f")
                r_mdo    = st.number_input("Retorno esperado del mercado", value=0.12, format="%.4f")
                beta_dcf = st.number_input("Beta del activo",
                                           value=fund["Beta"] if not pd.isna(fund["Beta"]) else 1.0,
                                           format="%.4f")
                kd       = st.number_input("Costo de la deuda kd (decimal)",        value=0.05, format="%.4f")
                w_equity = st.number_input("Peso del equity (decimal)",              value=0.80, format="%.4f")
                acciones = st.number_input("Acciones en circulación",                value=1e9,  format="%.0f")
            calcular_dcf = st.button("Calcular Valor Intrínseco (DCF)")

        if calcular_dcf and uo > 0:
            ke   = rf_dcf + beta_dcf * (r_mdo - rf_dcf)
            kd_d = kd * (1 - tasa_imp)
            wacc = ke * w_equity + kd_d * (1 - w_equity)

            uodi = uo * (1 - tasa_imp)
            fcf  = uodi + dda - delta_ktno - capex

            g_crec = 0.07
            g_term = 0.025
            flujos   = [fcf * (1 + g_crec)**t for t in range(1, 6)]
            vp_flujos = sum(f / (1 + wacc)**t for t, f in enumerate(flujos, 1))
            val_term  = flujos[-1] * (1 + g_term) / (wacc - g_term) if wacc > g_term else 0
            vp_term   = val_term / (1 + wacc)**5
            equity_val = vp_flujos + vp_term
            po_dcf     = equity_val / acciones if acciones > 0 else 0

            precio_mdo   = fund.get("precio") or 0
            potencial_dcf = (po_dcf / precio_mdo - 1) if precio_mdo > 0 else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("WACC",         f"{wacc*100:.2f}%")
            c2.metric("FCF calculado", f"${fcf:,.0f}")
            c3.metric("PO por DCF",    f"${po_dcf:.2f}")
            c4.metric("Potencial",     f"{potencial_dcf*100:+.1f}%")

            score_fundamental = float(np.clip(potencial_dcf * 2, -1, 1))
            st.write(f"**Score fundamental (DCF):** {score_fundamental:.4f}")
        else:
            pe = fund.get("P/E")
            if pe and not pd.isna(pe):
                score_fundamental = 0.5 if pe < 15 else (-0.5 if pe > 30 else 0.0)
            st.info(f"ℹ️ Sin datos DCF. Score aproximado por P/E: {score_fundamental:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # RECOMENDACIÓN FINAL
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("✅ Recomendación Final Ponderada")
    rec = recomendacion_final(
        score_tecnico, score_estadistico, score_fundamental,
        peso_tec, peso_est, peso_fund,
    )

    st.markdown(f"""
| Tipo de Análisis     | Score       | Peso   | Ponderado |
|----------------------|-------------|--------|-----------|
| Análisis Técnico     | {rec['score_tec']:+.4f} | {peso_tec*100:.0f}% | {rec['score_tec']*peso_tec/max(suma_pesos,0.01):+.4f} |
| Análisis Estadístico | {rec['score_est']:+.4f} | {peso_est*100:.0f}% | {rec['score_est']*peso_est/max(suma_pesos,0.01):+.4f} |
| Análisis Fundamental | {rec['score_fund']:+.4f} | {peso_fund*100:.0f}% | {rec['score_fund']*peso_fund/max(suma_pesos,0.01):+.4f} |
| **SCORE FINAL**      | **{rec['score']:+.4f}** | 100% | |
""")

    color_bg = {"green": "#1a4d1a", "red": "#4d1a1a", "orange": "#4d3a00"}
    st.markdown(
        f"""
        <div style='background-color:{color_bg[rec["color"]]};
                    border-left:6px solid {rec["color"]};
                    padding:20px; border-radius:8px; margin-top:10px;'>
            <h2 style='color:{rec["color"]}; margin:0;'>{rec["recomendacion"]}</h2>
            <p style='color:#ddd; margin:8px 0 0 0;'>{rec["descripcion"]}</p>
            <p style='color:#aaa; font-size:0.85rem; margin:4px
