
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
# FUNCIÓN: Valoración fundamental básica (Acciones)
# ─────────────────────────────────────────────────────────────────────────────
def valoracion_fundamental_basica(ticker):
    try:
        info = yf.Ticker(ticker).info
    except:
        info = {}

    return {
        "nombre":    info.get("longName", ticker),
        "sector":    info.get("sector", "N/D"),
        "precio":    info.get("currentPrice", np.nan),
        "P/E":       info.get("trailingPE", np.nan),
        "P/B":       info.get("priceToBook", np.nan),
        "EV/EBITDA": info.get("enterpriseToEbitda", np.nan),
        "ROE":       info.get("returnOnEquity", np.nan),
        "Beta":      info.get("beta", np.nan),
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

# ========================= PARTE 2 / 10 =========================
# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Descargar precios robusta
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

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Rendimientos logarítmicos
# ─────────────────────────────────────────────────────────────────────────────
def calcular_rendimientos(precios: pd.DataFrame) -> pd.DataFrame:
    return np.log(precios / precios.shift(1)).dropna()

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Índice base 100
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

# ========================= PARTE 3 / 10 =========================
# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE OPTIMIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def portafolio_markowitz(rendimientos: pd.DataFrame, rf: float) -> dict:
    n      = rendimientos.shape[1]
    medias = rendimientos.mean() * DIAS_ANIO
    cov    = rendimientos.cov() * DIAS_ANIO

    def varianza(w):
        return float(w @ cov.values @ w)

    res = minimize(
        varianza,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"ftol": 1e-9, "maxiter": 1000}
    )

    w   = res.x
    r   = float(w @ medias.values)
    v   = float(np.sqrt(w @ cov.values @ w))
    sr  = (r - rf) / v if v > 0 else 0.0
    return {"pesos": w, "rendimiento": r, "volatilidad": v, "sharpe": sr}


def portafolio_maximo_sharpe(rendimientos: pd.DataFrame, rf: float) -> dict:
    n      = rendimientos.shape[1]
    medias = rendimientos.mean() * DIAS_ANIO
    cov    = rendimientos.cov() * DIAS_ANIO

    def neg_sharpe(w):
        r = float(w @ medias.values)
        v = float(np.sqrt(w @ cov.values @ w))
        return -(r - rf) / (v + 1e-12)

    res = minimize(
        neg_sharpe,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"ftol": 1e-9, "maxiter": 1000}
    )

    w  = res.x
    r  = float(w @ medias.values)
    v  = float(np.sqrt(w @ cov.values @ w))
    sr = (r - rf) / v if v > 0 else 0.0
    return {"pesos": w, "rendimiento": r, "volatilidad": v, "sharpe": sr}


def portafolio_montecarlo(rendimientos: pd.DataFrame, rf: float, n_sim: int = 5000) -> dict:
    n      = rendimientos.shape[1]
    medias = rendimientos.mean() * DIAS_ANIO
    cov    = rendimientos.cov() * DIAS_ANIO

    mejor_sr = -np.inf
    mejor_w  = np.ones(n) / n

    for _ in range(n_sim):
        w  = np.random.dirichlet(np.ones(n))
        r  = float(w @ medias.values)
        v  = float(np.sqrt(w @ cov.values @ w))
        sr = (r - rf) / (v + 1e-12)
        if sr > mejor_sr:
            mejor_sr = sr
            mejor_w  = w

    r = float(mejor_w @ medias.values)
    v = float(np.sqrt(mejor_w @ cov.values @ mejor_w))
    return {"pesos": mejor_w, "rendimiento": r, "volatilidad": v, "sharpe": mejor_sr}

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Simulación Monte Carlo de precios
# ─────────────────────────────────────────────────────────────────────────────
def montecarlo_precio(rend_serie: pd.Series, precio_actual: float,
                      n_sim: int = 1000, horizonte: int = 252) -> pd.DataFrame:
    r = rend_serie.dropna()
    mu    = float(r.mean())
    sigma = float(r.std())
    Z     = np.random.standard_normal((horizonte, n_sim))
    tray  = precio_actual * np.exp(np.cumsum((mu - 0.5 * sigma**2) + sigma * Z, axis=0))
    return pd.DataFrame(tray)

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Indicadores técnicos
# ─────────────────────────────────────────────────────────────────────────────
def calcular_indicadores_tecnicos(precios: pd.Series) -> pd.DataFrame:
    s  = precios.dropna()
    df = pd.DataFrame({"Precio": s})

    for p in [5, 10, 20, 200]:
        df[f"MA{p}"] = df["Precio"].rolling(p).mean()

    delta = df["Precio"].diff()
    g = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    p = (-delta).clip(lower=0).ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + g / (p + 1e-12)))

    ema12         = df["Precio"].ewm(span=12, adjust=False).mean()
    ema26         = df["Precio"].ewm(span=26, adjust=False).mean()
    df["MACD"]    = ema12 - ema26
    df["Signal"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"]    = df["MACD"] - df["Signal"]

    return df

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Niveles Fibonacci
# ─────────────────────────────────────────────────────────────────────────────
def niveles_fibonacci(p_min: float, p_max: float) -> dict:
    r = p_max - p_min
    return {
        "Fibo 0%":    p_min,
        "Fibo 23.6%": p_min + 0.236 * r,
        "Fibo 38.2%": p_min + 0.382 * r,
        "Fibo 50%":   p_min + 0.500 * r,
        "Fibo 61.8%": p_min + 0.618 * r,
        "Fibo 100%":  p_max,
    }

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Valoración estadística
# ─────────────────────────────────────────────────────────────────────────────
def valoracion_estadistica(precios_ticker: pd.Series, precios_bmk: pd.Series,
                            anios_reg: float = 1.0) -> dict:
    df = pd.DataFrame({"ticker": precios_ticker, "bmk": precios_bmk}).dropna()
    if len(df) < 20:
        return None

    df_reg     = df.tail(int(anios_reg * DIAS_ANIO))
    ln_t       = np.log(df_reg["ticker"])
    ln_b       = np.log(df_reg["bmk"])
    X          = sm.add_constant(ln_b)
    modelo     = sm.OLS(ln_t, X).fit()
    alfa, beta = float(modelo.params.iloc[0]), float(modelo.params.iloc[1])
    r2         = float(modelo.rsquared)

    ln_bmk_act  = np.log(float(df["bmk"].iloc[-1]))
    precio_obj  = float(np.exp(alfa + beta * ln_bmk_act))
    precio_act  = float(df["ticker"].iloc[-1])
    potencial   = precio_obj / precio_act - 1

    vals = df["ticker"].dropna().values
    percentiles = {p: float(np.nanpercentile(vals, p * 100))
                   for p in [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]}
    pct_actual  = float(stats.percentileofscore(vals, precio_act)) / 100

    return {
        "alfa": alfa, "beta": beta, "r2": r2,
        "precio_obj_reg": precio_obj, "potencial_reg": potencial,
        "percentiles": percentiles, "percentil_actual": pct_actual,
        "precio_actual": precio_act, "confiable": r2 > 0.5,
    }

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Señal técnica
# ─────────────────────────────────────────────────────────────────────────────
def senal_tecnica(ind: pd.DataFrame) -> dict:
    """
    Evalúa los indicadores técnicos y asigna un score entre -1 (bajista) y +1 (alcista).

    Corrección Fibonacci:
    ─────────────────────
    La señal Fibonacci NO puede ser simplemente "precio < fibo 50%".
    Esa regla ignora la tendencia y produce señales invertidas (como el caso
    EEM: tendencia alcista, precio en máximos → bajista incorrecto).

    Lógica correcta:
      1. Detectar tendencia con pendiente lineal sobre todo el período.
      2. ALCISTA: medir qué tan poco ha retrocedido el precio desde el máximo.
         → Retroceso pequeño (precio cerca del máx) = señal alcista fuerte.
         → Retroceso profundo (precio lejos del máx) = señal bajista.
         Umbral: si retrocedió < 38.2% del rango → alcista (+1); > 61.8% → bajista (-1)
      3. BAJISTA: medir qué tan poco ha rebotado el precio desde el mínimo.
         → Rebote pequeño (precio cerca del mín) = señal bajista fuerte.
         → Rebote profundo (precio lejos del mín) = señal alcista.
         Umbral: si rebotó > 61.8% del rango → alcista (+1); < 38.2% → bajista (-1)
    """
    ult   = ind.iloc[-1]
    p_act = float(ult["Precio"])

    # ── Medias Móviles (Golden/Death Cross MA20 vs MA200) ─────────────────────
    ma20  = ult["MA20"]
    ma200 = ult["MA200"]
    if pd.isna(ma200):
        s_ma = 0.0
    else:
        s_ma = 1.0 if ma20 > ma200 else -1.0

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi = float(ult["RSI"])
    s_rsi = -1.0 if rsi > 70 else (1.0 if rsi < 30 else 0.0)

    # ── MACD ──────────────────────────────────────────────────────────────────
    s_macd = 1.0 if ult["MACD"] > ult["Signal"] else -1.0

    # ── Fibonacci con detección de tendencia ──────────────────────────────────
    p_min   = float(ind["Precio"].min())
    p_max   = float(ind["Precio"].max())
    rango   = p_max - p_min

    # Detectar tendencia mediante regresión lineal sobre los precios
    x_reg      = np.arange(len(ind))
    pendiente  = float(np.polyfit(x_reg, ind["Precio"].values, 1)[0])
    es_alcista = pendiente >= 0

    fibo = niveles_fibonacci(p_min, p_max)

    if rango > 0:
        if es_alcista:
            # ── Tendencia ALCISTA ──────────────────────────────────────────
            # El swing completo fue de mín → máx.
            # El retroceso se mide desde el máximo hacia abajo.
            # retroceso_ratio = 0   → precio exactamente en el máximo (muy alcista)
            # retroceso_ratio = 1   → precio exactamente en el mínimo (muy bajista)
            retroceso_ratio = (p_max - p_act) / rango   # entre 0 y 1
            if retroceso_ratio <= 0.382:
                # Retrocedió poco (< 38.2%): tendencia alcista intacta → señal ALCISTA
                s_fib = 1.0
            elif retroceso_ratio >= 0.618:
                # Retrocedió profundamente (> 61.8%): tendencia en riesgo → señal BAJISTA
                s_fib = -1.0
            else:
                # Zona intermedia (38.2% – 61.8%): señal NEUTRAL o leve
                s_fib = round(0.5 - retroceso_ratio, 1)   # aprox 0 en zona media
        else:
            # ── Tendencia BAJISTA ──────────────────────────────────────────
            # El swing completo fue de máx → mín.
            # El rebote se mide desde el mínimo hacia arriba.
            # rebote_ratio = 0   → precio exactamente en el mínimo (muy bajista)
            # rebote_ratio = 1   → precio exactamente en el máximo (fuerte rebote)
            rebote_ratio = (p_act - p_min) / rango   # entre 0 y 1
            if rebote_ratio >= 0.618:
                # Rebotó fuerte (> 61.8%): posible cambio de tendencia → señal ALCISTA
                s_fib = 1.0
            elif rebote_ratio <= 0.382:
                # Rebote débil (< 38.2%): sigue bajista → señal BAJISTA
                s_fib = -1.0
            else:
                # Zona intermedia
                s_fib = round(rebote_ratio - 0.5, 1)
    else:
        s_fib = 0.0   # Sin rango suficiente → neutral

    # Score ponderado: Fibonacci tiene el mayor peso (50%) por su relevancia técnica
    score = (1/6)*s_ma + (1/6)*s_rsi + (1/6)*s_macd + (1/2)*s_fib

    # Etiqueta descriptiva de la señal Fibonacci para la tabla
    fib_desc = (
        f"{'Alcista' if es_alcista else 'Bajista'} | "
        f"{'retroceso' if es_alcista else 'rebote'}: "
        f"{(retroceso_ratio if es_alcista else rebote_ratio)*100:.1f}% del rango"
        if rango > 0 else "Sin rango"
    )

    return {
        "señales": {
            "Medias Móviles": s_ma,
            "RSI":            s_rsi,
            "MACD":           s_macd,
            "Fibonacci":      s_fib,
        },
        "score_tecnico":  score,
        "rsi_valor":      rsi,
        "fibo_niveles":   fibo,
        "precio_actual":  p_act,
        "fib_tendencia":  "Alcista" if es_alcista else "Bajista",
        "fib_desc":       fib_desc,
        "fib_retroceso":  retroceso_ratio if (rango > 0 and es_alcista) else None,
        "fib_rebote":     rebote_ratio    if (rango > 0 and not es_alcista) else None,
    }

# ========================= PARTE 4 / 10 =========================
# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Recomendación final ponderada
# ─────────────────────────────────────────────────────────────────────────────
def recomendacion_final(s_tec, s_est, s_fund, p_tec, p_est, p_fund) -> dict:
    """
    Score ponderado: > 0.2 → COMPRA | -0.2 a 0.2 → MANTENER | < -0.2 → VENTA
    """
    total = p_tec + p_est + p_fund
    if total <= 0:
        total = 1.0
    score = (s_tec * p_tec + s_est * p_est + s_fund * p_fund) / total

    if score > 0.2:
        rec, color = "🟢 COMPRA",   "green"
        desc = "La acción parece infravalorada respecto a los criterios analizados."
    elif score < -0.2:
        rec, color = "🔴 VENTA",    "red"
        desc = "La acción parece sobrevalorada respecto a los criterios analizados."
    else:
        rec, color = "🟡 MANTENER", "orange"
        desc = "La acción cotiza cerca de su valor justo estimado."

    return {
        "score": round(score, 4), "recomendacion": rec,
        "color": color, "descripcion": desc,
        "score_tec": round(s_tec, 4),
        "score_est": round(s_est, 4),
        "score_fund": round(s_fund, 4),
    }

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: Tabla comparativa de portafolios
# ─────────────────────────────────────────────────────────────────────────────
def tabla_comparacion_portafolios(tickers, mk, ms, mc) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "Ticker":           t,
            "Peso Markowitz":   f"{mk['pesos'][i]*100:.1f}%",
            "Peso CAPM/Sharpe": f"{ms['pesos'][i]*100:.1f}%",
            "Peso Montecarlo":  f"{mc['pesos'][i]*100:.1f}%",
        })
    for label, key in [("Rendimiento EA", "rendimiento"),
                       ("Volatilidad EA",  "volatilidad"),
                       ("Sharpe Ratio",    "sharpe")]:
        fmt = (lambda v: f"{v*100:.2f}%") if key != "sharpe" else (lambda v: f"{v:.4f}")
        rows.append({
            "Ticker":           label,
            "Peso Markowitz":   fmt(mk[key]),
            "Peso CAPM/Sharpe": fmt(ms[key]),
            "Peso Montecarlo":  fmt(mc[key]),
        })
    return pd.DataFrame(rows)

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
# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: descarga de índices globales para el panel Macro
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def obtener_indices_globales() -> dict:
    """
    Descarga el precio actual y la variación diaria (%) de 6 índices/activos
    representativos del mercado global. TTL = 1 hora para no sobrecargar la API.
    Tickers:
      ^GSPC  = S&P 500            ^IXIC  = NASDAQ Composite
      ^DJI   = Dow Jones          ^COLCAP = Colcap Colombia
      GC=F   = Oro (futuros)      CL=F   = Petróleo WTI (futuros)
    """
    INDICES = {
        "S&P 500":   "^GSPC",
        "Dow Jones": "^DJI",
        "NASDAQ":    "^IXIC",
        "Colcap":    "^COLCAP",
        "Oro":       "GC=F",
        "WTI":       "CL=F",
    }
    resultados = {}
    for nombre, ticker_idx in INDICES.items():
        try:
            # Descargamos los últimos 5 días para calcular la variación diaria
            tmp = yf.download(ticker_idx, period="5d", auto_adjust=True, progress=False)
            # Extraer serie Close compatible con cualquier versión de yfinance
            if isinstance(tmp.columns, pd.MultiIndex):
                close = tmp["Close"].squeeze()
            else:
                close = tmp["Close"] if "Close" in tmp.columns else tmp.iloc[:, 0]
            close = pd.to_numeric(close, errors="coerce").dropna()
            if len(close) >= 2:
                ultimo  = float(close.iloc[-1])
                anterior = float(close.iloc[-2])
                cambio_pct = (ultimo / anterior - 1) * 100
                resultados[nombre] = {"precio": ultimo, "cambio": cambio_pct, "ticker": ticker_idx}
            elif len(close) == 1:
                resultados[nombre] = {"precio": float(close.iloc[0]), "cambio": 0.0, "ticker": ticker_idx}
        except Exception:
            resultados[nombre] = {"precio": np.nan, "cambio": np.nan, "ticker": ticker_idx}
    return resultados


with tabs[0]:
    st.header("🌍 Indicadores Macroeconómicos & Resumen de Tickers")

    # ── Fila 1: TRM + índices globales ────────────────────────────────────────
    st.subheader("📡 Indicadores de mercado en tiempo real")

    with st.spinner("Obteniendo índices globales..."):
        trm      = obtener_trm_colombia()
        indices  = obtener_indices_globales()

    # TRM en su propia métrica prominente
    col_trm, col_sep = st.columns([1, 5])
    with col_trm:
        if not np.isnan(trm):
            st.metric("💱 TRM COP/USD", f"{trm:,.2f}", help="Fuente: datos.gov.co")
        else:
            st.metric("💱 TRM COP/USD", "N/D")

    # Índices globales: una métrica por columna
    st.markdown("**Índices & Commodities**")
    cols_idx = st.columns(len(indices))
    for col_i, (nombre, datos) in zip(cols_idx, indices.items()):
        precio  = datos["precio"]
        cambio  = datos["cambio"]
        ticker_idx = datos["ticker"]
        if np.isnan(precio):
            col_i.metric(nombre, "N/D", help=ticker_idx)
        else:
            delta_str = f"{cambio:+.2f}%" if not np.isnan(cambio) else None
            col_i.metric(
                label=nombre,
                value=f"{precio:,.2f}",
                delta=delta_str,
                help=f"Ticker: {ticker_idx}",
            )

    st.divider()

    # ── Fila 2: Clasificación de tickers ─────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Clasificación de Tickers del Portafolio")
        resumen = [clasificar_ticker(t) for t in tickers_raw]
        df_resumen = pd.DataFrame(resumen)
        st.dataframe(df_resumen, use_container_width=True)

    with col2:
        st.subheader("📊 Variación diaria de índices globales")
        # Mini gráfico de barras horizontal con los cambios %
        nombres_idx = list(indices.keys())
        cambios_idx = [indices[n]["cambio"] for n in nombres_idx]
        colores_idx = ["#2ecc71" if c >= 0 else "#e74c3c" for c in cambios_idx]

        fig_idx = go.Figure(go.Bar(
            x=cambios_idx,
            y=nombres_idx,
            orientation="h",
            marker_color=colores_idx,
            text=[f"{c:+.2f}%" if not np.isnan(c) else "N/D" for c in cambios_idx],
            textposition="outside",
        ))
        fig_idx.update_layout(
            title="Variación diaria (%)",
            xaxis_title="Cambio %",
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=60, t=40, b=10),
        )
        fig_idx.add_vline(x=0, line_color="white", line_width=1)
        st.plotly_chart(fig_idx, use_container_width=True)

# ========================= PARTE 5 / 10 =========================
# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1: ANÁLISIS DE PORTAFOLIO
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("📊 Análisis del Portafolio")

    # 1.1 Base 100
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

    # 1.2 Rendimientos acumulados
    st.subheader("Rendimientos históricos acumulados")
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

    # 1.3 Métricas anualizadas
    st.subheader("Métricas de rendimiento y riesgo (anualizadas)")
    met_df = metricas_anuales(rendimientos_full, rf_diaria)
    st.dataframe(met_df, use_container_width=True)

    # 1.4 Beta, VaR, CVaR
    if benchmark_ok:
        st.subheader(f"Métricas de riesgo vs. benchmark ({benchmark})")
        tabla_riesgo = metricas_riesgo(rendimientos_full, benchmark)
        if not tabla_riesgo.empty:
            st.dataframe(tabla_riesgo, use_container_width=True)

    # 1.5 Correlación
    st.subheader("Matriz de correlación")
    corr = rendimientos_port.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlación entre activos del portafolio",
        template="plotly_dark",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # 1.6 Comparación vs benchmark
    if benchmark_ok and len(tickers_ok) >= 1:
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
        fig_vs.update_layout(title="Portafolio igual peso vs. Benchmark",
                             yaxis_tickformat=".0%", template="plotly_dark")
        st.plotly_chart(fig_vs, use_container_width=True)

# ========================= PARTE 6 / 10 =========================
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

# ========================= PARTE 7 / 10 =========================
# ═════════════════════════════════════════════════════════════════════════════
#  TAB 4: VALORACIÓN TICKER
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🔍 Valoración de Ticker Individual")

    ticker_val = st.selectbox("Selecciona el ticker a valorar", options=tickers_ok, key="val_sel")

    # ── Detectar tipo de activo para ajustar pesos por defecto ──────────────
    @st.cache_data(show_spinner=False)
    def _detectar_tipo(ticker_sym: str) -> str:
        """Devuelve 'ETF', 'Acción' u 'Otro' consultando yfinance.info."""
        try:
            info = yf.Ticker(ticker_sym).info
            qt   = str(info.get("quoteType", "")).upper()
            if qt == "ETF" or info.get("fundFamily"):
                return "ETF"
            elif qt == "EQUITY":
                return "Acción"
            return qt or "Otro"
        except Exception:
            return "Otro"

    tipo_val = _detectar_tipo(ticker_val)

    # Si es ETF (u otro sin estados financieros) → fundamental = 0, técnico 50, estadístico 50
    # Si es acción → distribuir 33/33/34 por defecto
    _es_sin_fundamentales = tipo_val in ("ETF", "INDEX", "FUTURE", "CURRENCY", "CRYPTOCURRENCY", "Otro")

    _def_tec  = 50 if _es_sin_fundamentales else 33
    _def_est  = 50 if _es_sin_fundamentales else 33
    _def_fund = 0  if _es_sin_fundamentales else 34

    st.subheader("🎛️ Ponderación de métodos")

    if _es_sin_fundamentales:
        st.info(
            f"ℹ️ **{ticker_val}** detectado como **{tipo_val}** — sin estados financieros. "
            "El análisis fundamental se fija en **0%** por defecto. "
            "Puedes ajustar los pesos manualmente."
        )
    else:
        st.caption("La suma debe ser 100%. Se normaliza automáticamente si no lo es.")

    c1, c2, c3 = st.columns(3)
    with c1:
        peso_tec  = st.slider(
            "📉 Análisis Técnico (%)", 0, 100, _def_tec,  step=1,
            key=f"sl_tec_{ticker_val}"
        ) / 100
    with c2:
        peso_est  = st.slider(
            "📐 Análisis Estadístico (%)", 0, 100, _def_est,  step=1,
            key=f"sl_est_{ticker_val}"
        ) / 100
    with c3:
        peso_fund = st.slider(
            "🏦 Análisis Fundamental (%)",
            0, 100,
            _def_fund,
            step=1,
            disabled=_es_sin_fundamentales,   # bloqueado si es ETF; el usuario puede desbloquearlo
            key=f"sl_fund_{ticker_val}",
            help="Desactivado para ETFs y activos sin estados financieros." if _es_sin_fundamentales else None,
        ) / 100

    # Si el slider fundamental está desactivado forzamos 0
    if _es_sin_fundamentales:
        peso_fund = 0.0

    suma_pesos = peso_tec + peso_est + peso_fund
    if abs(suma_pesos - 1.0) > 0.01:
        st.warning(f"⚠️ Los pesos suman {suma_pesos*100:.1f}%. Se normalizarán automáticamente.")

    # ─────────────────────────────────────────────────────────────────────────
    # ANÁLISIS TÉCNICO
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📉 1. Valoración Técnica")

    precios_t = precios_full[ticker_val]
    ind_df    = calcular_indicadores_tecnicos(precios_t)
    sen       = senal_tecnica(ind_df)

    # ── Gráfico A: Medias Móviles + RSI + MACD ───────────────────────────────
    st.markdown("##### 📈 Medias Móviles, RSI y MACD")
    fig_tec = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=[
            f"{ticker_val} – Precio y Medias Móviles",
            "RSI (14)", "MACD (12-26-9)"
        ]
    )

    # Precio + MAs
    fig_tec.add_trace(
        go.Scatter(x=ind_df.index, y=ind_df["Precio"],
                   name="Precio", line=dict(color="white", width=1)),
        row=1, col=1
    )
    for ma, color in [("MA5","cyan"),("MA10","yellow"),("MA20","orange"),("MA200","red")]:
        fig_tec.add_trace(
            go.Scatter(x=ind_df.index, y=ind_df[ma],
                       name=ma, line=dict(color=color, width=1.2)),
            row=1, col=1
        )

    # RSI
    fig_tec.add_trace(
        go.Scatter(x=ind_df.index, y=ind_df["RSI"],
                   name="RSI", line=dict(color="violet")),
        row=2, col=1
    )
    fig_tec.add_hline(y=70, line_dash="dash", line_color="red",    row=2, col=1,
                      annotation_text="Sobrecompra (70)", annotation_font_size=9)
    fig_tec.add_hline(y=50, line_dash="dot",  line_color="gray",   row=2, col=1)
    fig_tec.add_hline(y=30, line_dash="dash", line_color="green",  row=2, col=1,
                      annotation_text="Sobreventa (30)", annotation_font_size=9)

    # MACD
    colores_hist = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ind_df["Hist"].fillna(0)]
    fig_tec.add_trace(
        go.Bar(x=ind_df.index, y=ind_df["Hist"],
               name="Histograma", marker_color=colores_hist, opacity=0.7),
        row=3, col=1
    )
    fig_tec.add_trace(
        go.Scatter(x=ind_df.index, y=ind_df["MACD"],
                   name="MACD", line=dict(color="#3498db", width=1.5)),
        row=3, col=1
    )
    fig_tec.add_trace(
        go.Scatter(x=ind_df.index, y=ind_df["Signal"],
                   name="Signal", line=dict(color="#f39c12", width=1.5)),
        row=3, col=1
    )
    fig_tec.update_layout(height=700, template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_tec, use_container_width=True)

    # ── Gráfico B: Fibonacci con rango ajustable y tendencia detectada ──────
    st.markdown("##### 🌀 Retrocesos de Fibonacci")
    st.caption(
        "Selecciona el rango de fechas para calcular el máximo y mínimo del swing. "
        "Los niveles se recalculan automáticamente según el rango y la tendencia detectada."
    )

    # ── Control de rango de fechas ────────────────────────────────────────────
    fechas_disponibles = ind_df.index.tolist()
    fecha_min_disp = fechas_disponibles[0].date() if hasattr(fechas_disponibles[0], "date") else fechas_disponibles[0]
    fecha_max_disp = fechas_disponibles[-1].date() if hasattr(fechas_disponibles[-1], "date") else fechas_disponibles[-1]

    col_fi, col_ff = st.columns(2)
    with col_fi:
        fib_fecha_ini = st.date_input(
            "📅 Fecha inicio Fibonacci",
            value=fecha_min_disp,
            min_value=fecha_min_disp,
            max_value=fecha_max_disp,
            key=f"fib_ini_{ticker_val}",
        )
    with col_ff:
        fib_fecha_fin = st.date_input(
            "📅 Fecha fin Fibonacci",
            value=fecha_max_disp,
            min_value=fecha_min_disp,
            max_value=fecha_max_disp,
            key=f"fib_fin_{ticker_val}",
        )

    # Filtrar precios al rango seleccionado
    mask_fib = (
        ind_df.index >= pd.Timestamp(fib_fecha_ini)
    ) & (
        ind_df.index <= pd.Timestamp(fib_fecha_fin)
    )
    df_fib_rango = ind_df.loc[mask_fib, "Precio"]

    if len(df_fib_rango) < 2:
        st.warning("⚠️ Rango muy corto. Selecciona un período más amplio.")
    else:
        # ── Detectar tendencia en el rango seleccionado ───────────────────────
        # Comparamos el precio al inicio vs al final del rango.
        # Tendencia alcista  → el movimiento va de mínimo → máximo (retrocesos desde el pico).
        # Tendencia bajista  → el movimiento va de máximo → mínimo (retrocesos desde el suelo).
        precio_inicio_fib = float(df_fib_rango.iloc[0])
        precio_fin_fib    = float(df_fib_rango.iloc[-1])
        p_max_fib = float(df_fib_rango.max())
        p_min_fib = float(df_fib_rango.min())

        # Detectamos tendencia usando la pendiente de una regresión lineal rápida
        x_reg = np.arange(len(df_fib_rango))
        pendiente = np.polyfit(x_reg, df_fib_rango.values, 1)[0]
        tendencia_alcista = pendiente >= 0

        # ── Lógica de Fibonacci según tendencia ──────────────────────────────
        # ALCISTA: el swing completo va de mínimo (soporte) a máximo (resistencia).
        #   → Los niveles son retrocesos desde el máximo hacia abajo.
        #   → Fibo 0% = máximo | Fibo 100% = mínimo
        # BAJISTA: el swing completo va de máximo (techo) a mínimo (suelo).
        #   → Los niveles son retrocesos desde el mínimo hacia arriba.
        #   → Fibo 0% = mínimo | Fibo 100% = máximo
        if tendencia_alcista:
            fib_origen = p_max_fib   # desde donde retrocede (resistencia)
            fib_destino = p_min_fib  # hasta donde puede llegar el retroceso (soporte)
            etiqueta_tendencia = "🟢 Alcista — retrocesos desde el máximo"
        else:
            fib_origen = p_min_fib   # desde donde rebota (soporte)
            fib_destino = p_max_fib  # hasta donde puede llegar el rebote (resistencia)
            etiqueta_tendencia = "🔴 Bajista — retrocesos desde el mínimo"

        rango_fib  = abs(fib_origen - fib_destino)
        RATIOS_FIB = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
        NOMBRES_FIB = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]

        # Para tendencia alcista los niveles van de máx hacia abajo; para bajista de mín hacia arriba
        if tendencia_alcista:
            niveles_fib_rango = {
                f"Fibo {n}": fib_origen - ratio * rango_fib
                for ratio, n in zip(RATIOS_FIB, NOMBRES_FIB)
            }
        else:
            niveles_fib_rango = {
                f"Fibo {n}": fib_origen + ratio * rango_fib
                for ratio, n in zip(RATIOS_FIB, NOMBRES_FIB)
            }

        # ── Colores y estilos ─────────────────────────────────────────────────
        FIBO_ESTILOS = {
            "Fibo 0%":    ("#95a5a6", "dot",     1.2),
            "Fibo 23.6%": ("#3498db", "dash",    1.5),
            "Fibo 38.2%": ("#2ecc71", "dashdot", 1.5),
            "Fibo 50%":   ("#f1c40f", "solid",   2.0),
            "Fibo 61.8%": ("#e67e22", "dashdot", 1.5),
            "Fibo 78.6%": ("#9b59b6", "dash",    1.5),
            "Fibo 100%":  ("#e74c3c", "dot",     1.2),
        }

        # ── Construir el gráfico ──────────────────────────────────────────────
        fig_fib = go.Figure()

        # Precio real del activo en el rango (línea sólida, visible, SIN fill)
        fig_fib.add_trace(go.Scatter(
            x=df_fib_rango.index,
            y=df_fib_rango.values,
            name="Precio",
            line=dict(color="white", width=2),
            mode="lines",
        ))

        # Niveles de Fibonacci como líneas dispersas (scatter) para que aparezcan
        # en la leyenda y respeten el eje Y del precio — más confiable que add_hline
        x_inicio = df_fib_rango.index[0]
        x_fin    = df_fib_rango.index[-1]
        for nombre_f, val_f in niveles_fib_rango.items():
            color_f, dash_f, width_f = FIBO_ESTILOS.get(nombre_f, ("#888888", "dot", 1.2))
            fig_fib.add_trace(go.Scatter(
                x=[x_inicio, x_fin],
                y=[val_f, val_f],
                mode="lines",
                name=f"{nombre_f}: {val_f:.2f}",
                line=dict(color=color_f, dash=dash_f, width=width_f),
                hovertemplate=f"{nombre_f}: {val_f:.2f}<extra></extra>",
            ))

        # Línea del precio actual (último dato del rango)
        p_actual_fib = float(df_fib_rango.iloc[-1])
        fig_fib.add_trace(go.Scatter(
            x=[x_inicio, x_fin],
            y=[p_actual_fib, p_actual_fib],
            mode="lines",
            name=f"◀ Actual: {p_actual_fib:.2f}",
            line=dict(color="#00e5ff", width=2.5, dash="solid"),
            hovertemplate=f"Precio actual: {p_actual_fib:.2f}<extra></extra>",
        ))

        # Zona sombreada entre los dos niveles clave (38.2% y 61.8%)
        y_382 = niveles_fib_rango.get("Fibo 38.2%", None)
        y_618 = niveles_fib_rango.get("Fibo 61.8%", None)
        if y_382 and y_618:
            y0_zona, y1_zona = sorted([y_382, y_618])
            fig_fib.add_hrect(
                y0=y0_zona, y1=y1_zona,
                fillcolor="rgba(241,196,15,0.07)",
                line_width=0,
            )

        # Anotación de tendencia detectada
        fig_fib.add_annotation(
            x=df_fib_rango.index[len(df_fib_rango)//4],
            y=p_max_fib,
            text=etiqueta_tendencia,
            showarrow=False,
            font=dict(size=11, color="#aaaaaa"),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#555",
        )

        # Máximo y mínimo del rango marcados con marcadores
        idx_max = df_fib_rango.idxmax()
        idx_min = df_fib_rango.idxmin()
        fig_fib.add_trace(go.Scatter(
            x=[idx_max], y=[p_max_fib],
            mode="markers+text",
            name="Máx rango",
            marker=dict(color="#e74c3c", size=10, symbol="triangle-up"),
            text=[f"  Máx: {p_max_fib:.2f}"],
            textposition="top right",
            textfont=dict(color="#e74c3c", size=9),
            showlegend=False,
        ))
        fig_fib.add_trace(go.Scatter(
            x=[idx_min], y=[p_min_fib],
            mode="markers+text",
            name="Mín rango",
            marker=dict(color="#2ecc71", size=10, symbol="triangle-down"),
            text=[f"  Mín: {p_min_fib:.2f}"],
            textposition="bottom right",
            textfont=dict(color="#2ecc71", size=9),
            showlegend=False,
        ))

        fig_fib.update_layout(
            title=f"Retrocesos de Fibonacci – {ticker_val}  |  {etiqueta_tendencia}",
            yaxis_title="Precio",
            xaxis_title="Fecha",
            template="plotly_dark",
            height=480,
            hovermode="x unified",
            legend=dict(
                orientation="v",
                x=1.01, y=1,
                bgcolor="rgba(0,0,0,0.5)",
                font=dict(size=10),
            ),
        )
        st.plotly_chart(fig_fib, use_container_width=True)

        # ── Tabla resumen de niveles ──────────────────────────────────────────
        fib_df = pd.DataFrame([
            {
                "Nivel": k,
                "Precio": f"{v:.2f}",
                "Posición vs precio actual": (
                    "🔴 Resistencia" if v > p_actual_fib + 0.01 else
                    ("✅ Soporte"     if v < p_actual_fib - 0.01 else "◀ Precio actual")
                ),
                "Distancia %": f"{(v / p_actual_fib - 1) * 100:+.2f}%",
            }
            for k, v in niveles_fib_rango.items()
        ])
        col_tabla, col_info = st.columns([2, 1])
        with col_tabla:
            st.dataframe(fib_df, use_container_width=True, hide_index=True)
        with col_info:
            st.markdown(f"""
**Rango analizado:**
- Inicio: `{fib_fecha_ini}`
- Fin: `{fib_fecha_fin}`
- **Máximo:** `{p_max_fib:.2f}`
- **Mínimo:** `{p_min_fib:.2f}`
- **Rango:** `{rango_fib:.2f}` ({rango_fib/p_min_fib*100:.1f}%)
- **Tendencia:** {etiqueta_tendencia}
""")

    # Tabla de señales con nota explicativa en Fibonacci
    filas_senales = []
    for k, v in sen["señales"].items():
        emoji  = "🟢 ALCISTA" if v > 0 else ("🔴 BAJISTA" if v < 0 else "⚪ NEUTRAL")
        nota   = ""
        if k == "Fibonacci":
            nota = sen.get("fib_desc", "")
        filas_senales.append({
            "Indicador":  k,
            "Señal":      emoji,
            "Puntuación": v,
            "Detalle":    nota,
        })
    senales_df = pd.DataFrame(filas_senales)
    st.dataframe(senales_df, use_container_width=True, hide_index=True)

    # Métricas de resumen del análisis técnico
    col_sc1, col_sc2, col_sc3 = st.columns(3)
    col_sc1.metric("Score técnico", f"{sen['score_tecnico']:+.4f}")
    col_sc2.metric(
        "Tendencia Fibonacci",
        sen.get("fib_tendencia", "N/D"),
        help="Detectada por regresión lineal sobre el período completo"
    )
    # Mostrar retroceso o rebote según tendencia
    if sen.get("fib_retroceso") is not None:
        col_sc3.metric(
            "Retroceso desde máx",
            f"{sen['fib_retroceso']*100:.1f}%",
            delta="OK < 38.2%" if sen["fib_retroceso"] <= 0.382 else
                  ("⚠️ > 61.8%" if sen["fib_retroceso"] >= 0.618 else "Zona media"),
        )
    elif sen.get("fib_rebote") is not None:
        col_sc3.metric(
            "Rebote desde mín",
            f"{sen['fib_rebote']*100:.1f}%",
            delta="✅ > 61.8%" if sen["fib_rebote"] >= 0.618 else
                  ("⚠️ < 38.2%" if sen["fib_rebote"] <= 0.382 else "Zona media"),
        )
    score_tecnico = sen["score_tecnico"]

# ========================= PARTE 8 / 10 =========================
# ─────────────────────────────────────────────────────────────────────────────
# 2. ANÁLISIS ESTADÍSTICO
# ─────────────────────────────────────────────────────────────────────────────
    st.subheader("📐 2. Valoración Estadística")
    score_estadistico = 0.0

    if benchmark_ok:
        anios_reg = st.slider("Horizonte de regresión (años)", 1, min(anios, 5), 1)

        est = valoracion_estadistica(
            precios_full[ticker_val],
            precios_full[benchmark],
            anios_reg
        )

        if est:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precio actual",      f"${est['precio_actual']:.2f}")
            c2.metric("PO Regresión",        f"${est['precio_obj_reg']:.2f}",
                       f"{est['potencial_reg']*100:+.2f}%")
            c3.metric("R²", f"{est['r2']:.4f}",
                       "Confiable" if est["confiable"] else "Baja conf.")
            c4.metric("Percentil actual",   f"{est['percentil_actual']*100:.1f}%")

            # Gráfico regresión
            ln_t_all = np.log(precios_full[ticker_val])
            ln_b_all = np.log(precios_full[benchmark])
            x_lin = np.linspace(float(ln_b_all.min()), float(ln_b_all.max()), 100)
            y_lin = est["alfa"] + est["beta"] * x_lin

            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(
                x=ln_b_all, y=ln_t_all, mode="markers",
                name="Obs.",
                marker=dict(color="cyan", size=3, opacity=0.4)
            ))
            fig_reg.add_trace(go.Scatter(
                x=x_lin, y=y_lin, mode="lines",
                name=f"Regresión (R²={est['r2']:.3f})",
                line=dict(color="orange")
            ))
            fig_reg.update_layout(
                title=f"Regresión LN {ticker_val} vs LN {benchmark}",
                xaxis_title=f"ln({benchmark})",
                yaxis_title=f"ln({ticker_val})",
                template="plotly_dark"
            )
            st.plotly_chart(fig_reg, use_container_width=True)

            # Percentiles
            perc_labels = [f"P{int(k*100)}" for k in est["percentiles"]]
            perc_vals   = list(est["percentiles"].values())
            fig_perc = go.Figure()
            fig_perc.add_trace(go.Bar(
                x=perc_labels, y=perc_vals,
                marker_color="steelblue"
            ))
            fig_perc.add_hline(
                y=est["precio_actual"], line_color="red", line_dash="dash",
                annotation_text=f"Precio actual: ${est['precio_actual']:.2f}"
            )
            fig_perc.update_layout(
                title="Percentiles históricos de precio",
                template="plotly_dark"
            )
            st.plotly_chart(fig_perc, use_container_width=True)

            # Score estadístico
            score_reg  = float(np.clip(est["potencial_reg"], -1, 1))
            score_perc = float(np.clip(1 - 2 * est["percentil_actual"], -1, 1))
            score_estadistico = 0.6 * score_reg + 0.4 * score_perc

            st.write(f"**Score estadístico:** {score_estadistico:.4f}")

        else:
            st.warning("Datos insuficientes para la regresión.")
    else:
        st.warning("Benchmark no disponible para valoración estadística.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ANÁLISIS FUNDAMENTAL (ETF o Acción)
# ─────────────────────────────────────────────────────────────────────────────
    st.subheader("🏦 3. Valoración Fundamental")

    with st.spinner("Descargando datos fundamentales..."):
        fund = valoracion_fundamental_general(ticker_val)

    # ETF
    if "AUM" in fund:
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

    # ACCIÓN
    else:
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

# ========================= PARTE 9 / 10 =========================
        if calcular_dcf and uo > 0:
            # Cálculo WACC
            ke   = rf_dcf + beta_dcf * (r_mdo - rf_dcf)
            kd_d = kd * (1 - tasa_imp)
            wacc = ke * w_equity + kd_d * (1 - w_equity)

            # FCF
            uodi = uo * (1 - tasa_imp)
            fcf  = uodi + dda - delta_ktno - capex

            # Supuestos
            g_crec = 0.07
            g_term = 0.025

            # Flujos 5 años
            flujos = [fcf * (1 + g_crec)**t for t in range(1, 6)]
            vp_flujos = sum(f / (1 + wacc)**t for t, f in enumerate(flujos, 1))

            # Valor terminal
            val_term = flujos[-1] * (1 + g_term) / (wacc - g_term) if wacc > g_term else 0
            vp_term  = val_term / (1 + wacc)**5

            # Equity value
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
            # Score por múltiplos si no hay DCF
            pe = fund.get("P/E")
            if pe and not pd.isna(pe):
                score_fundamental = 0.5 if pe < 15 else (-0.5 if pe > 30 else 0.0)
            st.info(f"ℹ️ Sin datos DCF. Score aproximado por P/E: {score_fundamental:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # RECOMENDACIÓN FINAL
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("✅ Recomendación Final Ponderada")

    rec = recomendacion_final(
        score_tecnico,
        score_estadistico,
        score_fundamental,
        peso_tec,
        peso_est,
        peso_fund,
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
            <p style='color:#aaa; font-size:0.85rem; margin:4px 0 0 0;'>
                Score final: <strong>{rec['score']:+.4f}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ========================= PARTE 10 / 10 =========================
# ═════════════════════════════════════════════════════════════════════════════
#  PIE DE PÁGINA
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    """
    <div style='text-align:center; color:#555; font-size:0.8rem;'>
        Analizador de Portafolios & Valoración · Desarrollado por <strong>Diego CR</strong><br>
        Resultados meramente informativos · No constituyen asesoría de inversión ·
        Datos: Yahoo Finance vía <code>yfinance</code>
    </div>
    """,
    unsafe_allow_html=True,
)
