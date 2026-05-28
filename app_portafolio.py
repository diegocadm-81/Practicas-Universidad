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
import io

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
    page_title="Analizador de Portafolios",   # Título en la pestaña del navegador
    page_icon="📈",                           # Ícono de la pestaña
    layout="wide",                            # Usa todo el ancho de la pantalla
    initial_sidebar_state="expanded",        # Barra lateral expandida por defecto
)

# ─────────────────────────────────────────────────────────────────────────────
# CRÉDITOS DEL AUTOR (aviso fijo en todas las páginas)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:right; color:#888; font-size:0.78rem; padding-bottom:4px;'>
        Desarrollado por <strong>Diego CR</strong> · Análisis de Portafolios & Valoración
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES Y PARÁMETROS GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
TASA_LIBRE_RIESGO = 0.0457   # Tasa libre de riesgo anualizada (US 10Y aprox.)
DIAS_ANIO = 252              # Días hábiles de mercado por año
PERIODOS_LABEL = {"1 Año": 1, "3 Años": 3, "5 Años": 5, "7 Años": 7, "10 Años": 10, "15 Años": 15, "20 Años": 20}  # Horizontes de análisis


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 – SIDEBAR: ingreso de tickers y configuración general
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Configuración")
    st.caption("Ingresa los tickers y parámetros del análisis")

    # Campo de texto para múltiples tickers del portafolio, separados por coma
    tickers_input = st.text_area(
        "Tickers del portafolio (separados por coma)",
        value="SPY, QQQ, IWM, EFA, EEM",
        help="Ejemplo: AAPL, MSFT, AMZN, GOOGL"
    )

    # Ticker del benchmark para comparar el portafolio
    benchmark_input = st.text_input(
        "Ticker del Benchmark",
        value="SPY",
        help="Índice de referencia. Ejemplo: ^GSPC para S&P 500"
    )

    # Horizonte de tiempo para la descarga de precios
    periodo = st.selectbox(
        "Período de análisis",
        options=list(PERIODOS_LABEL.keys()),
        index=2,
        help="Ventana histórica de precios a descargar"
    )

    # Tasa libre de riesgo anualizada (editable por el usuario)
    rf = st.number_input(
        "Tasa libre de riesgo (anual, decimal)",
        value=TASA_LIBRE_RIESGO,
        step=0.001,
        format="%.4f",
        help="Ejemplo: 0.0457 = 4.57%"
    )

    # Botón que dispara la descarga y todo el análisis
    btn_analizar = st.button("🚀 Analizar Portafolio", type="primary", use_container_width=True)

    st.divider()
    st.caption("**Autor:** Diego CR")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: descargar y limpiar precios de cierre ajustados
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def descargar_precios(tickers: list, benchmark: str, anios: int) -> pd.DataFrame:
    """
    Descarga precios de cierre ajustados desde Yahoo Finance para todos los
    tickers del portafolio más el benchmark.

    Parámetros
    ----------
    tickers   : lista de símbolos del portafolio
    benchmark : símbolo del índice de referencia
    anios     : años hacia atrás a descargar

    Retorna
    -------
    DataFrame con columnas = tickers + benchmark, índice = fecha,
    sin filas con NaN (filas donde algún ticker no tenga precio).
    """
    fecha_fin    = datetime.date.today()
    fecha_inicio = fecha_fin - datetime.timedelta(days=int(anios * 365.25))

    # Descarga masiva de todos los tickers en una sola llamada a yfinance
    todos = list(dict.fromkeys(tickers + [benchmark]))   # preserva orden, sin duplicados

    raw = yf.download(
        todos,
        start=str(fecha_inicio),
        end=str(fecha_fin),
        auto_adjust=True,    # Precios ya ajustados por dividendos y splits
        progress=False,
        group_by="ticker",   # yfinance >=0.2: agrupa por ticker para tener MultiIndex claro
    )

    # ── Normalización del resultado de yfinance ───────────────────────────────
    # yfinance puede devolver distintas estructuras según versión y número de tickers:
    #   - MultiIndex (ticker, campo): cuando hay >1 ticker con group_by="ticker"
    #   - MultiIndex (campo, ticker): cuando hay >1 ticker sin group_by
    #   - Plano: cuando hay exactamente 1 ticker

    if isinstance(raw.columns, pd.MultiIndex):
        # Intentamos extraer nivel "Close" de cualquiera de los dos posibles ordenamientos
        nivel_0 = raw.columns.get_level_values(0).unique().tolist()
        nivel_1 = raw.columns.get_level_values(1).unique().tolist()

        if "Close" in nivel_0:
            # Estructura (campo, ticker) → raw["Close"] da columnas = tickers
            precios = raw["Close"].copy()
        elif "Close" in nivel_1:
            # Estructura (ticker, campo) → necesitamos re-ordenar
            precios = raw.xs("Close", axis=1, level=1).copy()
        else:
            # Fallback: descarga individual ticker por ticker
            dfs = {}
            for t in todos:
                try:
                    tmp = yf.download(t, start=str(fecha_inicio), end=str(fecha_fin),
                                      auto_adjust=True, progress=False)
                    if not tmp.empty:
                        # En versiones recientes el resultado puede ser MultiIndex incluso para 1 ticker
                        if isinstance(tmp.columns, pd.MultiIndex):
                            dfs[t] = tmp["Close"].squeeze()
                        else:
                            dfs[t] = tmp["Close"]
                except Exception:
                    pass
            precios = pd.DataFrame(dfs)
    else:
        # Un solo ticker: columnas planas
        if "Close" in raw.columns:
            precios = raw[["Close"]].rename(columns={"Close": todos[0]})
        else:
            precios = raw.iloc[:, :1].copy()
            precios.columns = [todos[0]]

    # Aseguramos que el índice se llame "Date" para consistencia con los gráficos
    precios.index.name = "Date"

    # Aplanamos columnas por si quedó algún nivel residual
    if isinstance(precios.columns, pd.MultiIndex):
        precios.columns = [str(c[0]) if isinstance(c, tuple) else str(c)
                           for c in precios.columns]

    # Convertimos a numérico (por si hubo strings o tipos mixtos)
    precios = precios.apply(pd.to_numeric, errors="coerce")

    # Eliminamos filas donde ALGÚN ticker no tenga precio
    # (criterio del Excel AP_Data: "eliminar filas donde algún ticker no tenga precio")
    precios = precios.dropna()

    return precios


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: calcular rendimientos diarios logarítmicos
# ─────────────────────────────────────────────────────────────────────────────
def calcular_rendimientos(precios: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los rendimientos diarios logarítmicos: r_t = ln(P_t / P_{t-1}).
    El logaritmo tiene ventajas: aditivo en el tiempo y mejor normalidad.
    La primera fila queda como NaN y se elimina.
    """
    return np.log(precios / precios.shift(1)).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: métricas de rendimiento y riesgo anualizadas
# ─────────────────────────────────────────────────────────────────────────────
def metricas_anuales(rendimientos: pd.DataFrame, rf_diaria: float) -> pd.DataFrame:
    """
    Para cada activo calcula:
      - Rendimiento EA (efectivo anual): exp(media_diaria * 252) - 1
      - Volatilidad anual: desv_std_diaria * sqrt(252)
      - Sharpe Ratio: (rendEA - rf) / volatilidad
    """
    media   = rendimientos.mean()          # Media de rendimientos diarios
    vol_d   = rendimientos.std()           # Desviación estándar diaria
    rend_ea = np.exp(media * DIAS_ANIO) - 1   # Rendimiento efectivo anual
    vol_ea  = vol_d * np.sqrt(DIAS_ANIO)      # Volatilidad anualizada

    # Sharpe Ratio: rendimiento excedente por unidad de riesgo
    sharpe  = (rend_ea - rf_diaria * DIAS_ANIO) / vol_ea

    tabla = pd.DataFrame({
        "Rendimiento EA": rend_ea,
        "Volatilidad EA": vol_ea,
        "Sharpe Ratio":   sharpe,
    }).T

    return tabla


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: índice base 100
# ─────────────────────────────────────────────────────────────────────────────
def base_100(precios: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """
    Normaliza todos los precios al mismo punto de partida (base).
    Fórmula: idx_t = (P_t / P_0) * base
    Permite comparar visualmente activos de distintas magnitudes de precio.

    Robustez: si el DataFrame viene vacío o la primera fila tiene ceros/NaN,
    devuelve el DataFrame tal cual para evitar IndexError / ZeroDivisionError.
    """
    if precios.empty or len(precios) == 0:
        return precios
    primer_valor = precios.iloc[0].replace(0, np.nan)   # Evita división por cero
    return (precios / primer_valor) * base


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE OPTIMIZACIÓN DE PORTAFOLIO
# ─────────────────────────────────────────────────────────────────────────────

def portafolio_markowitz(rendimientos: pd.DataFrame, rf: float) -> dict:
    """
    Optimización media-varianza de Markowitz.
    Encuentra los pesos que minimizan la varianza del portafolio
    sujeto a que los pesos sumen 1 y sean no negativos (sin ventas en corto).

    Retorna diccionario con pesos, rendimiento EA, volatilidad y Sharpe.
    """
    n      = rendimientos.shape[1]                       # Número de activos
    medias = rendimientos.mean() * DIAS_ANIO             # Rendimientos medios diarios anualizados
    cov    = rendimientos.cov() * DIAS_ANIO              # Matriz de covarianza anualizada

    # Función objetivo: varianza del portafolio w' Σ w
    def varianza(w):
        return w @ cov.values @ w

    # Restricción: suma de pesos = 1 (portafolio completamente invertido)
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Límites: cada peso entre 0% y 100% (sin ventas en corto)
    limites = [(0, 1)] * n

    # Pesos iniciales: distribución uniforme (igual peso a todos)
    w0 = np.ones(n) / n

    # Minimización de varianza usando SLSQP (Sequential Least Squares Programming)
    resultado = minimize(varianza, w0, method="SLSQP",
                         bounds=limites, constraints=restricciones,
                         options={"ftol": 1e-9, "maxiter": 1000})

    w_opt = resultado.x                            # Pesos óptimos
    rend  = float(w_opt @ medias.values)           # Rendimiento esperado del portafolio
    vol   = float(np.sqrt(w_opt @ cov.values @ w_opt))  # Volatilidad del portafolio
    sharpe = (rend - rf) / vol                     # Ratio de Sharpe

    return {"pesos": w_opt, "rendimiento": rend, "volatilidad": vol, "sharpe": sharpe}


def portafolio_maximo_sharpe(rendimientos: pd.DataFrame, rf: float) -> dict:
    """
    Maximiza el Ratio de Sharpe (Máximo Information Ratio).
    Equivalente al portafolio de tangencia en la frontera eficiente.
    Se usa para la optimización 'CAPM-inspired' (máx rendimiento ajustado por riesgo).
    """
    n      = rendimientos.shape[1]
    medias = rendimientos.mean() * DIAS_ANIO
    cov    = rendimientos.cov() * DIAS_ANIO

    # Función objetivo: negativo del Sharpe (minimizamos el negativo = maximizamos)
    def neg_sharpe(w):
        rend = w @ medias.values
        vol  = np.sqrt(w @ cov.values @ w)
        return -(rend - rf) / (vol + 1e-12)   # +epsilon evita división por cero

    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    limites = [(0, 1)] * n
    w0 = np.ones(n) / n

    resultado = minimize(neg_sharpe, w0, method="SLSQP",
                         bounds=limites, constraints=restricciones,
                         options={"ftol": 1e-9, "maxiter": 1000})

    w_opt  = resultado.x
    rend   = float(w_opt @ medias.values)
    vol    = float(np.sqrt(w_opt @ cov.values @ w_opt))
    sharpe = (rend - rf) / vol

    return {"pesos": w_opt, "rendimiento": rend, "volatilidad": vol, "sharpe": sharpe}


def portafolio_montecarlo(rendimientos: pd.DataFrame, rf: float, n_sim: int = 5000) -> dict:
    """
    Simulación de Monte Carlo: genera 'n_sim' portafolios con pesos aleatorios
    y selecciona el que maximiza el Sharpe Ratio.

    Ventaja: explora el espacio de pesos sin asumir convexidad; captura
    combinaciones que los métodos analíticos podrían omitir.
    """
    n      = rendimientos.shape[1]
    medias = rendimientos.mean() * DIAS_ANIO
    cov    = rendimientos.cov() * DIAS_ANIO

    mejor_sharpe = -np.inf     # Guardamos el mejor Sharpe encontrado
    mejor_pesos  = None

    for _ in range(n_sim):
        # Generamos pesos aleatorios de distribución uniforme Dirichlet
        w = np.random.dirichlet(np.ones(n))
        rend = w @ medias.values
        vol  = np.sqrt(w @ cov.values @ w)
        sr   = (rend - rf) / (vol + 1e-12)

        # Actualizamos si este portafolio supera el mejor Sharpe hasta ahora
        if sr > mejor_sharpe:
            mejor_sharpe = sr
            mejor_pesos  = w

    rend_final = float(mejor_pesos @ medias.values)
    vol_final  = float(np.sqrt(mejor_pesos @ cov.values @ mejor_pesos))

    return {
        "pesos":       mejor_pesos,
        "rendimiento": rend_final,
        "volatilidad": vol_final,
        "sharpe":      mejor_sharpe,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: métricas de riesgo Beta, VaR y CVaR
# ─────────────────────────────────────────────────────────────────────────────
def metricas_riesgo(rendimientos: pd.DataFrame, benchmark: str, rf: float,
                    nivel: float = 0.05) -> pd.DataFrame:
    """
    Calcula para cada activo:
      - Beta: sensibilidad del activo respecto al mercado (benchmark)
      - VaR (95%): pérdida máxima esperada con 95% de confianza en 1 día
      - CVaR (95%): pérdida media en el peor 5% de los días (Expected Shortfall)

    Parámetros
    ----------
    nivel : nivel de significancia (0.05 = 95% confianza)
    """
    activos = [c for c in rendimientos.columns if c != benchmark]
    r_bmk   = rendimientos[benchmark]    # Rendimientos del benchmark

    resultados = {}
    for ticker in activos:
        r = rendimientos[ticker]

        # ── Beta ─────────────────────────────────────────────────────────────
        # Beta = Cov(activo, benchmark) / Var(benchmark)
        cov_ab = np.cov(r, r_bmk)[0, 1]
        var_bmk = np.var(r_bmk)
        beta = cov_ab / var_bmk if var_bmk != 0 else np.nan

        # ── VaR histórico (percentil) ─────────────────────────────────────────
        # El 'nivel'% peor rendimiento observado en la muestra
        var_hist = np.percentile(r, nivel * 100)

        # ── CVaR / Expected Shortfall ─────────────────────────────────────────
        # Media de los rendimientos que caen por debajo del VaR
        cvar = r[r <= var_hist].mean()

        resultados[ticker] = {
            "Beta":       round(beta, 4),
            "VaR 95% (1d)": f"{var_hist*100:.2f}%",
            "CVaR 95% (1d)": f"{cvar*100:.2f}%",
        }

    return pd.DataFrame(resultados).T


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: simulación Monte Carlo de precio futuro (1 año)
# ─────────────────────────────────────────────────────────────────────────────
def montecarlo_precio(rendimientos_ticker: pd.Series, precio_actual: float,
                      n_sim: int = 1000, horizonte: int = 252) -> pd.DataFrame:
    """
    Simula 'n_sim' trayectorias de precio para un horizonte dado usando
    movimiento browniano geométrico (GBM).

    GBM: S_{t+1} = S_t * exp((mu - 0.5*sigma²)*dt + sigma*sqrt(dt)*Z)
    donde Z ~ N(0,1) es un choque aleatorio estándar.

    Retorna DataFrame con los rendimientos acumulados de cada simulación.
    """
    mu    = rendimientos_ticker.mean()    # Drift diario (media log-rendimiento)
    sigma = rendimientos_ticker.std()     # Volatilidad diaria

    # Matriz de choques aleatorios N(0,1): (horizonte x n_sim)
    Z = np.random.standard_normal((horizonte, n_sim))

    # Rendimientos diarios simulados
    r_sim = (mu - 0.5 * sigma**2) + sigma * Z

    # Precio simulado: precio_actual * exp(suma acumulada de rendimientos)
    trayectorias = precio_actual * np.exp(np.cumsum(r_sim, axis=0))

    return pd.DataFrame(trayectorias)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: indicadores técnicos
# ─────────────────────────────────────────────────────────────────────────────
def calcular_indicadores_tecnicos(precios: pd.Series) -> pd.DataFrame:
    """
    Calcula sobre la serie de precios:
      - MA5, MA10, MA20, MA200: Medias móviles simples
      - RSI(14): Índice de fuerza relativa
      - MACD y Signal(9): Media móvil de convergencia/divergencia
      - Niveles de Fibonacci: 23.6%, 38.2%, 50%, 61.8%, 100%
    """
    df = pd.DataFrame({"Precio": precios})

    # ── Medias móviles simples ────────────────────────────────────────────────
    for p in [5, 10, 20, 200]:
        df[f"MA{p}"] = df["Precio"].rolling(p).mean()

    # ── RSI (14 días) ─────────────────────────────────────────────────────────
    # RSI = 100 - (100 / (1 + RS)), donde RS = media_ganancias / media_pérdidas
    delta    = df["Precio"].diff()
    ganancias = delta.clip(lower=0)       # Solo días positivos
    perdidas  = (-delta).clip(lower=0)    # Solo días negativos (en positivo)
    media_g  = ganancias.ewm(span=14, adjust=False).mean()   # EMA 14 de ganancias
    media_p  = perdidas.ewm(span=14, adjust=False).mean()    # EMA 14 de pérdidas
    rs       = media_g / (media_p + 1e-12)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD ──────────────────────────────────────────────────────────────────
    # MACD = EMA(12) - EMA(26); Signal = EMA(9) del MACD; Histograma = MACD - Signal
    ema12       = df["Precio"].ewm(span=12, adjust=False).mean()
    ema26       = df["Precio"].ewm(span=26, adjust=False).mean()
    df["MACD"]  = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"]   = df["MACD"] - df["Signal"]

    return df


def niveles_fibonacci(precio_min: float, precio_max: float) -> dict:
    """
    Calcula los niveles de retroceso de Fibonacci.
    Fibonacci mide hasta dónde puede retroceder un movimiento antes de continuar.
    Niveles clásicos: 23.6%, 38.2%, 50%, 61.8%, 100%
    """
    rango = precio_max - precio_min
    return {
        "Fibo 0% (mín)":    precio_min,
        "Fibo 23.6%":       precio_min + 0.236 * rango,
        "Fibo 38.2%":       precio_min + 0.382 * rango,
        "Fibo 50%":         precio_min + 0.500 * rango,
        "Fibo 61.8%":       precio_min + 0.618 * rango,
        "Fibo 100% (máx)":  precio_max,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: valoración estadística (regresión lineal + percentiles)
# ─────────────────────────────────────────────────────────────────────────────
def valoracion_estadistica(precios_ticker: pd.Series, precios_bmk: pd.Series,
                            anios_regresion: float = 1.0) -> dict:
    """
    Dos métodos estadísticos de valoración:

    1. Regresión lineal sobre precios LN:
       ln(P_ticker) = alfa + beta * ln(P_benchmark) + epsilon
       El precio objetivo es exp(alfa + beta * ln(P_bmk_actual))

    2. Percentiles históricos del precio:
       Informa en qué percentil está el precio actual respecto a su historia.
    """
    # Alineamos las series temporalmente
    df = pd.DataFrame({"ticker": precios_ticker, "bmk": precios_bmk}).dropna()

    # ── Filtro por horizonte de regresión ─────────────────────────────────────
    dias_reg   = int(anios_regresion * DIAS_ANIO)
    df_reg     = df.tail(dias_reg)   # Últimos N días para la regresión

    # ── Transformación logarítmica de precios ─────────────────────────────────
    ln_ticker  = np.log(df_reg["ticker"])    # ln(precio ticker)
    ln_bmk     = np.log(df_reg["bmk"])       # ln(precio benchmark)

    # ── Regresión OLS: ln(ticker) ~ alfa + beta * ln(bmk) ────────────────────
    X          = sm.add_constant(ln_bmk)     # Agrega columna de unos para el intercepto
    modelo     = sm.OLS(ln_ticker, X).fit()  # Ordinary Least Squares
    alfa       = modelo.params.iloc[0]       # Intercepto
    beta       = modelo.params.iloc[1]       # Pendiente
    r2         = modelo.rsquared             # Coeficiente de determinación

    # Precio teórico usando el ln del benchmark actual
    ln_bmk_actual = np.log(df["bmk"].iloc[-1])
    precio_obj    = np.exp(alfa + beta * ln_bmk_actual)
    precio_actual = df["ticker"].iloc[-1]
    potencial     = (precio_obj / precio_actual) - 1   # Potencial de valorización

    # ── Percentiles sobre toda la historia ───────────────────────────────────
    percentiles_vals = {
        p: np.percentile(df["ticker"], p * 100)
        for p in [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    }
    # ¿En qué percentil está el precio actual?
    percentil_actual = stats.percentileofscore(df["ticker"], precio_actual) / 100

    return {
        "alfa":           alfa,
        "beta":           beta,
        "r2":             r2,
        "precio_obj_reg": precio_obj,
        "potencial_reg":  potencial,
        "percentiles":    percentiles_vals,
        "percentil_actual": percentil_actual,
        "precio_actual":  precio_actual,
        "precio_bmk":     df["bmk"].iloc[-1],
        "confiable":      r2 > 0.5,   # R² > 50% = estadísticamente confiable
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: señal técnica (score entre -1 y 1)
# ─────────────────────────────────────────────────────────────────────────────
def senal_tecnica(ind: pd.DataFrame) -> dict:
    """
    Evalúa cada indicador técnico y asigna puntuación de -1 (bajista) a +1 (alcista).
    Luego pondera según los pesos del modelo (Fibonacci 50%, resto 16.67% c/u).
    """
    ult = ind.iloc[-1]    # Última fila del DataFrame de indicadores

    senales = {}

    # ── Medias Móviles ────────────────────────────────────────────────────────
    # Golden Cross (MA50 > MA200) = señal alcista (+1)
    # Death Cross  (MA50 < MA200) = señal bajista (-1)
    ma50  = ind["MA20"].iloc[-1]     # Usamos MA20 como proxy de corto plazo
    ma200 = ind["MA200"].iloc[-1] if not pd.isna(ind["MA200"].iloc[-1]) else np.nan
    if not np.isnan(ma200):
        senales["Medias Móviles"] = 1.0 if ma50 > ma200 else -1.0
    else:
        senales["Medias Móviles"] = 0.0

    # ── RSI ───────────────────────────────────────────────────────────────────
    # RSI > 70 → sobrecompra → señal de venta (-1)
    # RSI < 30 → sobreventa  → señal de compra (+1)
    # RSI entre 30-70 → neutral (0)
    rsi = ult["RSI"]
    if rsi > 70:
        senales["RSI"] = -1.0
    elif rsi < 30:
        senales["RSI"] = 1.0
    else:
        senales["RSI"] = 0.0

    # ── MACD ──────────────────────────────────────────────────────────────────
    # MACD > Signal → momentum alcista (+1)
    # MACD < Signal → momentum bajista (-1)
    senales["MACD"] = 1.0 if ult["MACD"] > ult["Signal"] else -1.0

    # ── Fibonacci ─────────────────────────────────────────────────────────────
    # El precio cerca de un soporte (retroceso profundo) → compra (+1)
    # El precio cerca de una resistencia (poca corrección) → venta (-1)
    p_min = ind["Precio"].min()
    p_max = ind["Precio"].max()
    fibo  = niveles_fibonacci(p_min, p_max)
    p_act = ult["Precio"]
    # Calculamos distancia relativa al centro (50% de Fibonacci)
    centro = fibo["Fibo 50%"]
    senales["Fibonacci"] = 1.0 if p_act < centro else -1.0

    # ── Pesos internos del análisis técnico ───────────────────────────────────
    pesos = {
        "Medias Móviles": 1/6,
        "RSI":            1/6,
        "MACD":           1/6,
        "Fibonacci":      1/2,   # Fibonacci tiene el doble de peso según el modelo
    }

    # Score ponderado técnico (entre -1 y +1)
    score_tecnico = sum(senales[k] * pesos[k] for k in senales)

    return {
        "señales":        senales,
        "score_tecnico":  score_tecnico,
        "rsi_valor":      rsi,
        "macd_valor":     ult["MACD"],
        "signal_valor":   ult["Signal"],
        "fibo_niveles":   fibo,
        "precio_actual":  p_act,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: valoración fundamental básica (múltiplos)
# ─────────────────────────────────────────────────────────────────────────────
def valoracion_fundamental_basica(ticker: str) -> dict:
    """
    Descarga métricas fundamentales básicas vía yfinance:
      - P/E (RPG), P/B (Q Tobin proxy), EV/EBITDA, ROE, Yield dividendo
    Compara el P/E del activo con el sector para dar señal direccional.

    Nota: Para una valoración FCF completa (como en el Excel Val Fundamental)
    se requieren estados financieros detallados (UO, Capex, KTNO, deuda, etc.)
    que el usuario debe ingresar manualmente en el formulario de la app.
    """
    t     = yf.Ticker(ticker)
    info  = t.info

    # Extraemos métricas con manejo de valores faltantes (None → NaN)
    pe       = info.get("trailingPE",        np.nan)
    pb       = info.get("priceToBook",       np.nan)
    ev_ebitda = info.get("enterpriseToEbitda", np.nan)
    roe      = info.get("returnOnEquity",    np.nan)
    div_yield = info.get("dividendYield",    np.nan)
    beta     = info.get("beta",              np.nan)
    precio   = info.get("currentPrice",      np.nan)
    nombre   = info.get("longName",          ticker)
    sector   = info.get("sector",            "N/D")
    mkt_cap  = info.get("marketCap",         np.nan)

    return {
        "nombre":    nombre,
        "sector":    sector,
        "precio":    precio,
        "P/E":       pe,
        "P/B":       pb,
        "EV/EBITDA": ev_ebitda,
        "ROE":       roe,
        "Yield":     div_yield,
        "Beta":      beta,
        "MktCap":    mkt_cap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: recomendación final ponderada (Buy / Hold / Sell)
# ─────────────────────────────────────────────────────────────────────────────
def recomendacion_final(score_tec: float, score_est: float, score_fund: float,
                        peso_tec: float, peso_est: float, peso_fund: float) -> dict:
    """
    Combina los tres scores con los pesos asignados por el usuario.

    Score entre -1 y +1:
      > +0.2  → COMPRA   (infravalorada / señal alcista)
      -0.2 a +0.2 → MANTENER (en precio justo)
      < -0.2  → VENTA    (sobrevalorada / señal bajista)

    Esta lógica replica la hoja 'Resumen Ponderación' del Excel.
    """
    # Normalizamos los pesos para que sumen exactamente 1
    total = peso_tec + peso_est + peso_fund
    if total <= 0:
        total = 1

    # Score ponderado final
    score = (score_tec * peso_tec + score_est * peso_est + score_fund * peso_fund) / total

    # Asignación de recomendación según umbrales
    if score > 0.2:
        rec   = "🟢 COMPRA"
        color = "green"
        desc  = "La acción parece infravalorada respecto a los criterios analizados."
    elif score < -0.2:
        rec   = "🔴 VENTA"
        color = "red"
        desc  = "La acción parece sobrevalorada respecto a los criterios analizados."
    else:
        rec   = "🟡 MANTENER"
        color = "orange"
        desc  = "La acción cotiza cerca de su valor justo estimado."

    return {
        "score":           round(score, 4),
        "recomendacion":   rec,
        "color":           color,
        "descripcion":     desc,
        "score_tec":       round(score_tec, 4),
        "score_est":       round(score_est, 4),
        "score_fund":      round(score_fund, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: resumen comparativo de los tres portafolios optimizados
# ─────────────────────────────────────────────────────────────────────────────
def tabla_comparacion_portafolios(tickers: list, mk: dict, ms: dict, mc: dict) -> pd.DataFrame:
    """
    Construye una tabla comparativa con los pesos y métricas
    de los tres métodos de optimización: Markowitz, Máx Sharpe (CAPM) y Monte Carlo.
    """
    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "Ticker":          t,
            "Peso Markowitz":  f"{mk['pesos'][i]*100:.1f}%",
            "Peso CAPM/Sharpe": f"{ms['pesos'][i]*100:.1f}%",
            "Peso Montecarlo": f"{mc['pesos'][i]*100:.1f}%",
        })

    # Fila de métricas del portafolio
    for label, d in [("Rendimiento EA", "rendimiento"),
                     ("Volatilidad EA", "volatilidad"),
                     ("Sharpe Ratio",   "sharpe")]:
        rows.append({
            "Ticker":           label,
            "Peso Markowitz":   f"{mk[d]*100:.2f}%" if d != "sharpe" else f"{mk[d]:.4f}",
            "Peso CAPM/Sharpe": f"{ms[d]*100:.2f}%" if d != "sharpe" else f"{ms[d]:.4f}",
            "Peso Montecarlo":  f"{mc[d]*100:.2f}%" if d != "sharpe" else f"{mc[d]:.4f}",
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  INICIO DEL FLUJO PRINCIPAL DE LA APLICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

# Mostramos pantalla de bienvenida mientras no se ha presionado el botón
if not btn_analizar:
    st.markdown("## 📈 Analizador de Portafolios & Valoración")
    st.markdown("""
    **Bienvenido.** Esta herramienta permite:

    - 📊 **Análisis de portafolio**: precios en base 100, correlaciones, rendimientos históricos
    - ⚙️ **Optimización**: Markowitz (mín varianza), máx Sharpe (CAPM) y Monte Carlo
    - 🎲 **Simulación Monte Carlo** de precios futuros (1000 trayectorias)
    - 📉 **Valoración técnica**: Medias móviles, RSI, MACD, Fibonacci
    - 📐 **Valoración estadística**: Regresión lineal y percentiles históricos
    - 🏦 **Valoración fundamental**: Múltiplos de mercado y FCF (entrada manual)
    - ✅ **Recomendación final** ponderada por el usuario (Comprar / Mantener / Vender)

    👈 **Ingresa los tickers en el panel izquierdo y haz clic en "Analizar Portafolio".**

    ---
    *Desarrollado por **Diego CR** · Los resultados son meramente informativos y no constituyen asesoría financiera.*
    """)
    st.stop()   # Detiene la ejecución si el usuario no ha presionado el botón


# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO PRINCIPAL (ejecuta al presionar el botón)
# ─────────────────────────────────────────────────────────────────────────────

# Parseo de los tickers ingresados por el usuario
tickers_raw = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
benchmark   = benchmark_input.strip().upper()
anios       = PERIODOS_LABEL[periodo]         # Convierte "3 Años" → 3
rf_diaria   = rf / DIAS_ANIO                  # Tasa libre de riesgo diaria

# Validación básica: debe haber al menos 2 tickers
if len(tickers_raw) < 2:
    st.error("❌ Por favor ingresa al menos 2 tickers para el portafolio.")
    st.stop()

# Aseguramos que el benchmark no esté duplicado en los tickers del portafolio
tickers = [t for t in tickers_raw if t != benchmark]

# ── Descarga de datos ─────────────────────────────────────────────────────────
with st.spinner("⏳ Descargando precios desde Yahoo Finance..."):
    try:
        precios_full = descargar_precios(tickers, benchmark, anios)
    except Exception as e:
        st.error(f"❌ Error al descargar datos: {e}")
        st.stop()

# Verificamos que todos los tickers hayan descargado datos
tickers_ok  = [t for t in tickers  if t in precios_full.columns]
benchmark_ok = benchmark in precios_full.columns

if len(tickers_ok) == 0:
    st.error("❌ No se pudo descargar datos para ninguno de los tickers. Verifica los símbolos.")
    st.stop()

if not benchmark_ok:
    st.warning(f"⚠️ No se encontraron datos para el benchmark '{benchmark}'. Se omitirá.")

# Precios solo del portafolio (sin benchmark)
precios_port = precios_full[tickers_ok]

# Cálculo de rendimientos diarios logarítmicos
rendimientos_full = calcular_rendimientos(precios_full)
rendimientos_port = rendimientos_full[tickers_ok]

# ─────────────────────────────────────────────────────────────────────────────
# TABS PRINCIPALES DE LA APLICACIÓN
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Análisis de Portafolio",
    "⚙️ Optimización",
    "🎲 Monte Carlo",
    "🔍 Valoración Ticker",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: ANÁLISIS DE PORTAFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Análisis del Portafolio")

    # ── 1.1 Precios en Base 100 ────────────────────────────────────────────────
    st.subheader("Precios normalizados (Base 100)")
    st.caption("Permite comparar el desempeño relativo de activos con distintos niveles de precio.")

    precios_b100  = base_100(precios_full, 100)     # Normalización a base 100
    fig_b100 = px.line(
        precios_b100.reset_index(),
        x="Date", y=precios_b100.columns.tolist(),
        title="Evolución de activos en base 100",
        labels={"value": "Valor (base 100)", "Date": "Fecha", "variable": "Ticker"},
        template="plotly_dark",
    )
    st.plotly_chart(fig_b100, use_container_width=True)

    # ── 1.2 Rendimientos históricos acumulados ─────────────────────────────────
    st.subheader("Rendimientos históricos acumulados")
    rend_acum = (1 + np.exp(rendimientos_full.cumsum()) - 1)  # Rendimiento acumulado
    fig_rend = px.line(
        rend_acum.reset_index(),
        x="Date", y=rend_acum.columns.tolist(),
        title="Rendimiento acumulado",
        labels={"value": "Rendimiento acumulado", "Date": "Fecha", "variable": "Ticker"},
        template="plotly_dark",
    )
    fig_rend.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_rend, use_container_width=True)

    # ── 1.3 Tabla de métricas anualizadas ─────────────────────────────────────
    st.subheader("Métricas de rendimiento y riesgo (anualizadas)")
    met = metricas_anuales(rendimientos_full, rf_diaria)
    # Formato porcentual para rendimiento y volatilidad
    # Construimos la tabla de formato como objeto string independiente
    # (pandas >=2 no permite asignar strings a columnas float in-place)
    met_fmt = pd.DataFrame({
        col: {
            "Rendimiento EA": "{:.2%}".format(met.loc["Rendimiento EA", col]),
            "Volatilidad EA":  "{:.2%}".format(met.loc["Volatilidad EA",  col]),
            "Sharpe Ratio":    "{:.4f}".format(met.loc["Sharpe Ratio",    col]),
        }
        for col in met.columns
    }).T
    st.dataframe(met_fmt, use_container_width=True)

    # ── 1.4 Métricas de riesgo: Beta, VaR, CVaR ───────────────────────────────
    if benchmark_ok:
        st.subheader(f"Métricas de riesgo vs. benchmark ({benchmark})")
        tabla_riesgo = metricas_riesgo(rendimientos_full, benchmark, rf)
        st.dataframe(tabla_riesgo, use_container_width=True)

    # ── 1.5 Matriz de correlación ──────────────────────────────────────────────
    st.subheader("Matriz de correlación")
    st.caption("Valores cercanos a 1: activos muy correlacionados. Cercanos a 0: independientes.")
    corr = rendimientos_port.corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlación entre activos del portafolio",
        template="plotly_dark",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── 1.6 Comparación contra benchmark ──────────────────────────────────────
    if benchmark_ok:
        st.subheader(f"Comparación portafolio igualitario vs. {benchmark}")
        # Portafolio de igual peso (equal weight)
        w_eq      = np.ones(len(tickers_ok)) / len(tickers_ok)
        rend_port_ew = rendimientos_port @ w_eq          # Rendimiento diario del portafolio EW
        rend_acum_port = (np.exp(rend_port_ew.cumsum()) - 1)   # Acumulado
        rend_acum_bmk  = (np.exp(rendimientos_full[benchmark].cumsum()) - 1)

        fig_vs = go.Figure()
        fig_vs.add_trace(go.Scatter(x=rend_acum_port.index, y=rend_acum_port,
                                    name="Portafolio (EW)", line=dict(color="cyan")))
        fig_vs.add_trace(go.Scatter(x=rend_acum_bmk.index, y=rend_acum_bmk,
                                    name=benchmark, line=dict(color="orange")))
        fig_vs.update_layout(title="Portafolio igual peso vs. Benchmark",
                             yaxis_tickformat=".0%", template="plotly_dark")
        st.plotly_chart(fig_vs, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: OPTIMIZACIÓN DE PORTAFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Optimización de Portafolio")
    st.info("Se presentan los tres métodos: Markowitz (mín varianza), CAPM / Máx Sharpe, y Monte Carlo.")

    with st.spinner("Optimizando portafolios..."):
        mk_res = portafolio_markowitz(rendimientos_port, rf)       # Método Markowitz
        ms_res = portafolio_maximo_sharpe(rendimientos_port, rf)   # Método Máx Sharpe
        mc_res = portafolio_montecarlo(rendimientos_port, rf)      # Método Monte Carlo

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    tabla_opt = tabla_comparacion_portafolios(tickers_ok, mk_res, ms_res, mc_res)
    st.subheader("Resumen comparativo de los tres portafolios óptimos")
    st.dataframe(tabla_opt, use_container_width=True)

    # ── Gráfico de barras comparativo de pesos ────────────────────────────────
    st.subheader("Distribución de pesos por método")
    n_tickers = len(tickers_ok)
    df_pesos = pd.DataFrame({
        "Ticker":        tickers_ok,
        "Markowitz":     mk_res["pesos"] * 100,
        "CAPM / Sharpe": ms_res["pesos"] * 100,
        "Montecarlo":    mc_res["pesos"] * 100,
    }).melt(id_vars="Ticker", var_name="Método", value_name="Peso (%)")

    fig_pesos = px.bar(
        df_pesos, x="Ticker", y="Peso (%)", color="Método",
        barmode="group", title="Pesos óptimos por método (%)",
        template="plotly_dark",
    )
    st.plotly_chart(fig_pesos, use_container_width=True)

    # ── Métricas detalladas por método ────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    for col, nombre, res in [
        (col1, "Markowitz (Mín Varianza)", mk_res),
        (col2, "CAPM / Máx Sharpe",        ms_res),
        (col3, "Monte Carlo",              mc_res),
    ]:
        with col:
            st.metric(f"📌 {nombre}", "", "")
            st.write(f"**Rendimiento EA:** {res['rendimiento']*100:.2f}%")
            st.write(f"**Volatilidad EA:** {res['volatilidad']*100:.2f}%")
            st.write(f"**Sharpe Ratio:**   {res['sharpe']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: SIMULACIÓN MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Simulación Monte Carlo de Precios (1 año)")
    st.caption("Simula 1000 trayectorias de precio usando movimiento browniano geométrico (GBM).")

    # El usuario selecciona el ticker a simular
    ticker_mc = st.selectbox("Selecciona el ticker a simular", options=tickers_ok)

    with st.spinner("Corriendo simulación..."):
        r_tick   = rendimientos_port[ticker_mc]         # Rendimientos históricos del ticker
        p_actual = float(precios_port[ticker_mc].iloc[-1])   # Último precio

        # Generamos 1000 trayectorias a 252 días vista
        sim = montecarlo_precio(r_tick, p_actual, n_sim=1000, horizonte=252)

    # ── Gráfico de trayectorias ───────────────────────────────────────────────
    fig_mc = go.Figure()
    # Graficamos todas las trayectorias con alta transparencia
    for col in sim.columns[:200]:   # Solo primeras 200 para no saturar el gráfico
        fig_mc.add_trace(go.Scatter(
            y=sim[col], mode="lines",
            line=dict(width=0.5, color="rgba(100,180,255,0.15)"),
            showlegend=False,
        ))

    # Percentiles relevantes como líneas de referencia
    p5  = sim.iloc[-1].quantile(0.05)    # Peor escenario esperado (5%)
    p50 = sim.iloc[-1].quantile(0.50)    # Escenario mediano (50%)
    p95 = sim.iloc[-1].quantile(0.95)    # Mejor escenario esperado (95%)

    fig_mc.add_hline(y=p5,  line_dash="dot", line_color="red",   annotation_text=f"P5: ${p5:.2f}")
    fig_mc.add_hline(y=p50, line_dash="dot", line_color="yellow",annotation_text=f"P50: ${p50:.2f}")
    fig_mc.add_hline(y=p95, line_dash="dot", line_color="green", annotation_text=f"P95: ${p95:.2f}")

    fig_mc.update_layout(
        title=f"Monte Carlo: {ticker_mc} – 1000 trayectorias a 252 días",
        yaxis_title="Precio simulado (USD)",
        xaxis_title="Días hábiles",
        template="plotly_dark",
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # ── Estadísticas de la distribución final ────────────────────────────────
    st.subheader("Distribución de precios al final de la simulación (día 252)")
    precios_finales = sim.iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Precio actual",     f"${p_actual:.2f}")
    col2.metric("P5  (bajista)",      f"${p5:.2f}",   f"{(p5/p_actual-1)*100:.1f}%")
    col3.metric("P50 (base)",         f"${p50:.2f}",  f"{(p50/p_actual-1)*100:.1f}%")
    col4.metric("P95 (alcista)",      f"${p95:.2f}",  f"{(p95/p_actual-1)*100:.1f}%")
    col5.metric("Media simulada",     f"${precios_finales.mean():.2f}")

    # Histograma de precios finales
    fig_hist = px.histogram(
        precios_finales, nbins=50,
        title=f"Distribución de precios finales simulados – {ticker_mc}",
        labels={"value": "Precio final (USD)"},
        template="plotly_dark",
        color_discrete_sequence=["#4FC3F7"],
    )
    fig_hist.add_vline(x=p_actual, line_dash="dash", line_color="white",
                       annotation_text="Precio actual")
    st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: VALORACIÓN DE TICKER INDIVIDUAL
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Valoración de Ticker Individual")

    # El usuario elige qué ticker quiere valorar en detalle
    ticker_val = st.selectbox(
        "Selecciona el ticker a valorar",
        options=tickers_ok,
        key="ticker_val",
    )

    # Pesos de cada tipo de valoración (configurables por el usuario)
    st.subheader("🎛️ Ponderación de métodos de valoración")
    st.caption("Asigna el peso que tendrá cada tipo de valoración en la recomendación final. "
               "La suma debe ser 100%.")

    c1, c2, c3 = st.columns(3)
    with c1:
        peso_tec  = st.slider("Peso Análisis Técnico (%)",     0, 100, 33, step=1) / 100
    with c2:
        peso_est  = st.slider("Peso Análisis Estadístico (%)", 0, 100, 33, step=1) / 100
    with c3:
        peso_fund = st.slider("Peso Análisis Fundamental (%)", 0, 100, 34, step=1) / 100

    # Aviso si los pesos no suman exactamente 100%
    suma_pesos = peso_tec + peso_est + peso_fund
    if abs(suma_pesos - 1.0) > 0.01:
        st.warning(f"⚠️ Los pesos suman {suma_pesos*100:.1f}%. Se normalizarán automáticamente.")

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB A: ANÁLISIS TÉCNICO
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("📉 1. Valoración Técnica")

    precios_t = precios_full[ticker_val]    # Precios del ticker seleccionado
    ind_df    = calcular_indicadores_tecnicos(precios_t)  # Indicadores técnicos
    sen       = senal_tecnica(ind_df)       # Evaluación de señales

    # Gráfico de velas con MAs y Fibonacci
    fig_tec = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             row_heights=[0.5, 0.25, 0.25],
                             subplot_titles=[f"{ticker_val} – Precio y Medias Móviles",
                                             "RSI (14)",
                                             "MACD y Signal"])

    # Precio y medias móviles
    fig_tec.add_trace(go.Scatter(x=ind_df.index, y=ind_df["Precio"],
                                  name="Precio", line=dict(color="white", width=1)), row=1, col=1)
    colores_ma = {"MA5": "cyan", "MA10": "yellow", "MA20": "orange", "MA200": "red"}
    for ma, color in colores_ma.items():
        fig_tec.add_trace(go.Scatter(x=ind_df.index, y=ind_df[ma],
                                      name=ma, line=dict(color=color, width=1.2)),
                           row=1, col=1)

    # Niveles de Fibonacci como líneas horizontales
    for nombre_fibo, valor_fibo in sen["fibo_niveles"].items():
        fig_tec.add_hline(y=valor_fibo, line_dash="dot",
                           line_color="rgba(255,215,0,0.4)",
                           annotation_text=nombre_fibo,
                           annotation_font_size=9, row=1, col=1)

    # RSI con zonas de sobrecompra/sobreventa
    fig_tec.add_trace(go.Scatter(x=ind_df.index, y=ind_df["RSI"],
                                  name="RSI", line=dict(color="violet")), row=2, col=1)
    fig_tec.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
    fig_tec.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig_tec.add_hline(y=50, line_dash="dot",  line_color="gray",  row=2, col=1)

    # MACD
    colores_hist = ["green" if v >= 0 else "red" for v in ind_df["Hist"]]
    fig_tec.add_trace(go.Bar(x=ind_df.index, y=ind_df["Hist"],
                              name="Histograma", marker_color=colores_hist), row=3, col=1)
    fig_tec.add_trace(go.Scatter(x=ind_df.index, y=ind_df["MACD"],
                                  name="MACD", line=dict(color="blue")), row=3, col=1)
    fig_tec.add_trace(go.Scatter(x=ind_df.index, y=ind_df["Signal"],
                                  name="Signal", line=dict(color="orange")), row=3, col=1)

    fig_tec.update_layout(height=700, template="plotly_dark", showlegend=True)
    st.plotly_chart(fig_tec, use_container_width=True)

    # Tabla de señales técnicas
    st.write("**Señales técnicas:**")
    senales_df = pd.DataFrame([
        {"Indicador": k, "Señal": "🟢 ALCISTA" if v > 0 else ("🔴 BAJISTA" if v < 0 else "⚪ NEUTRAL"),
         "Puntuación": v}
        for k, v in sen["señales"].items()
    ])
    senales_df["Score Técnico"] = ""
    senales_df.at[len(senales_df)-1, "Score Técnico"] = f"{sen['score_tecnico']:.4f}"
    st.dataframe(senales_df, use_container_width=True)

    score_tecnico = sen["score_tecnico"]

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB B: ANÁLISIS ESTADÍSTICO
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("📐 2. Valoración Estadística")

    # Requiere benchmark disponible
    if benchmark_ok:
        anios_reg = st.slider(
            "Horizonte de regresión (años)", 1, min(anios, 5), 1,
            help="Período histórico usado para estimar alfa y beta de la regresión"
        )

        est = valoracion_estadistica(precios_full[ticker_val],
                                      precios_full[benchmark],
                                      anios_regresion=anios_reg)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precio actual",      f"${est['precio_actual']:.2f}")
        col2.metric("PO Regresión Lineal", f"${est['precio_obj_reg']:.2f}",
                    f"{est['potencial_reg']*100:+.2f}%")
        col3.metric("R² de la regresión",  f"{est['r2']:.4f}",
                    "✅ Confiable" if est["confiable"] else "⚠️ Baja confiabilidad")
        col4.metric("Percentil actual",    f"{est['percentil_actual']*100:.1f}%")

        # Gráfico de dispersión LN ticker vs LN benchmark (regresión)
        ln_t  = np.log(precios_full[ticker_val])
        ln_b  = np.log(precios_full[benchmark])
        x_lin = np.linspace(ln_b.min(), ln_b.max(), 100)
        y_lin = est["alfa"] + est["beta"] * x_lin

        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(x=ln_b, y=ln_t, mode="markers",
                                      name="Observaciones",
                                      marker=dict(color="cyan", size=3, opacity=0.4)))
        fig_reg.add_trace(go.Scatter(x=x_lin, y=y_lin, mode="lines",
                                      name=f"Regresión (R²={est['r2']:.3f})",
                                      line=dict(color="orange")))
        fig_reg.update_layout(
            title=f"Regresión LN {ticker_val} vs LN {benchmark}",
            xaxis_title=f"ln({benchmark})", yaxis_title=f"ln({ticker_val})",
            template="plotly_dark",
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        # Gráfico de percentiles
        perc_vals  = list(est["percentiles"].values())
        perc_labels = [f"P{int(k*100)}" for k in est["percentiles"].keys()]
        fig_perc = go.Figure()
        fig_perc.add_trace(go.Bar(x=perc_labels, y=perc_vals,
                                   marker_color="steelblue", name="Percentil"))
        fig_perc.add_hline(y=est["precio_actual"], line_color="red", line_dash="dash",
                            annotation_text=f"Precio actual: ${est['precio_actual']:.2f}")
        fig_perc.update_layout(title="Distribución de percentiles históricos de precio",
                                template="plotly_dark")
        st.plotly_chart(fig_perc, use_container_width=True)

        # Score estadístico: combinación regresión (60%) + percentil (40%)
        # Señal de regresión: entre -1 y +1 según el potencial de valorización
        score_reg  = np.clip(est["potencial_reg"], -1, 1)   # Acotado a [-1, 1]
        # Señal de percentil: si está en percentil alto (>70%) señal de venta; bajo (<30%) compra
        p_act      = est["percentil_actual"]
        score_perc = np.clip(1 - 2 * p_act, -1, 1)  # P_act=1 → -1 (venta); P_act=0 → +1 (compra)

        score_estadistico = 0.6 * score_reg + 0.4 * score_perc

        st.write(f"**Score estadístico:** {score_estadistico:.4f} "
                 f"(Regresión: {score_reg:.4f} × 60% | Percentil: {score_perc:.4f} × 40%)")
    else:
        st.warning("⚠️ Benchmark no disponible. La valoración estadística no puede calcularse.")
        score_estadistico = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB C: ANÁLISIS FUNDAMENTAL
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🏦 3. Valoración Fundamental")

    with st.spinner("Descargando datos fundamentales..."):
        fund = valoracion_fundamental_basica(ticker_val)

    # Ficha del activo
    st.markdown(f"**{fund['nombre']}** | Sector: {fund['sector']}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Precio mercado", f"${fund['precio']:.2f}" if fund['precio'] else "N/D")
    col2.metric("P/E (RPG)",      f"{fund['P/E']:.1f}x"   if fund['P/E'] else "N/D")
    col3.metric("P/B (Q Tobin)",  f"{fund['P/B']:.2f}x"   if fund['P/B'] else "N/D")
    col4.metric("EV/EBITDA",      f"{fund['EV/EBITDA']:.1f}x" if fund['EV/EBITDA'] else "N/D")
    col5.metric("ROE",            f"{fund['ROE']*100:.1f}%" if fund['ROE'] else "N/D")

    # ── Formulario de FCF manual (replicando hoja Val Fundamental del Excel) ─
    st.markdown("#### Valoración por DCF (Flujo de Caja Libre Descontado)")
    st.caption("Ingresa los datos financieros del reporte más reciente para calcular el valor intrínseco.")

    with st.expander("📋 Ingresar datos financieros para DCF (opcional)"):
        col1, col2 = st.columns(2)
        with col1:
            uo       = st.number_input("Utilidad Operativa (EBIT) 12M",       value=0.0, format="%.0f")
            dda      = st.number_input("Depreciación y Amortización 12M",     value=0.0, format="%.0f")
            capex    = st.number_input("CAPEX 12M",                            value=0.0, format="%.0f")
            delta_ktno = st.number_input("Variación KTNO",                    value=0.0, format="%.0f")
            tasa_imp = st.number_input("Tasa de impuestos (decimal)",          value=0.25, format="%.4f")
        with col2:
            rf_dcf   = st.number_input("Tasa libre de riesgo",                 value=rf, format="%.4f")
            r_mdo    = st.number_input("Retorno esperado del mercado",          value=0.12, format="%.4f")
            beta_dcf = st.number_input("Beta del activo",                       value=fund.get("Beta", 1.0) or 1.0, format="%.4f")
            kd       = st.number_input("Costo de la deuda (kd, decimal)",       value=0.05, format="%.4f")
            w_equity = st.number_input("Peso del equity en capital (decimal)",  value=0.80, format="%.4f")
            acciones = st.number_input("Acciones en circulación",               value=1e9,  format="%.0f")

        calcular_dcf = st.button("Calcular Valor Intrínseco (DCF)")

    score_fundamental = 0.0   # Por defecto neutro si no se calcula

    if calcular_dcf and uo > 0:
        # ── Costo del equity (CAPM): Ke = rf + beta * (rm - rf) ──────────────
        ke   = rf_dcf + beta_dcf * (r_mdo - rf_dcf)
        kd_d = kd * (1 - tasa_imp)                     # Kd después de impuestos
        wacc = ke * w_equity + kd_d * (1 - w_equity)  # WACC ponderado

        # ── Free Cash Flow ──────────────────────────────────────────────────
        uodi = uo * (1 - tasa_imp)                     # Utilidad operativa después de impuestos
        fcf  = uodi + dda - delta_ktno - capex         # FCF según modelo del Excel

        # ── Proyección 5 años + valor terminal ───────────────────────────────
        g_crec = 0.07    # Tasa de crecimiento asumida para los flujos proyectados
        g_term = 0.025   # Tasa de crecimiento a perpetuidad (conservadora)

        flujos = [fcf * (1 + g_crec)**t for t in range(1, 6)]   # 5 años proyectados

        # Valor presente de flujos del período relevante
        vp_flujos = sum(f / (1 + wacc)**t for t, f in enumerate(flujos, start=1))

        # Valor terminal (Gordon Growth Model)
        val_term = flujos[-1] * (1 + g_term) / (wacc - g_term) if wacc > g_term else 0
        vp_term  = val_term / (1 + wacc)**5       # Descuento al presente

        # Equity value y precio objetivo por acción
        equity_val  = vp_flujos + vp_term
        precio_obj_dcf = equity_val / acciones if acciones > 0 else 0

        precio_mdo   = fund.get("precio", 0) or 0
        potencial_dcf = (precio_obj_dcf / precio_mdo - 1) if precio_mdo > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("WACC",           f"{wacc*100:.2f}%")
        col2.metric("FCF calculado",  f"${fcf:,.0f}")
        col3.metric("PO por DCF",     f"${precio_obj_dcf:.2f}")
        col4.metric("Potencial",      f"{potencial_dcf*100:+.1f}%")

        # Score fundamental basado en potencial DCF
        score_fundamental = np.clip(potencial_dcf * 2, -1, 1)
        st.write(f"**Score fundamental (DCF):** {score_fundamental:.4f}")

    # Si no se ingresaron datos DCF, usamos múltiplos básicos como aproximación
    if score_fundamental == 0.0:
        pe = fund.get("P/E")
        if pe and not np.isnan(pe):
            # P/E < 15 → infravalorado (+); P/E > 30 → sobrevalorado (-); entre → neutro
            if pe < 15:
                score_fundamental = 0.5
            elif pe > 30:
                score_fundamental = -0.5
            else:
                score_fundamental = 0.0
        st.info(f"ℹ️ Sin datos DCF completos. Score fundamental aproximado por P/E: {score_fundamental:.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # RECOMENDACIÓN FINAL PONDERADA
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("✅ Recomendación Final Ponderada")

    rec = recomendacion_final(
        score_tec=score_tecnico,
        score_est=score_estadistico,
        score_fund=score_fundamental,
        peso_tec=peso_tec,
        peso_est=peso_est,
        peso_fund=peso_fund,
    )

    # Tabla de matriz de ponderación (replica hoja 'Resumen Ponderación' del Excel)
    st.markdown(f"""
    | Tipo de Análisis         | Score (-1 a 1)               | Peso  | Score Ponderado              |
    |--------------------------|------------------------------|-------|------------------------------|
    | Análisis Técnico         | {rec['score_tec']:+.4f}       | {peso_tec*100:.0f}%   | {rec['score_tec']*peso_tec/suma_pesos:+.4f} |
    | Análisis Estadístico     | {rec['score_est']:+.4f}       | {peso_est*100:.0f}%   | {rec['score_est']*peso_est/suma_pesos:+.4f} |
    | Análisis Fundamental     | {rec['score_fund']:+.4f}      | {peso_fund*100:.0f}%  | {rec['score_fund']*peso_fund/suma_pesos:+.4f} |
    | **SCORE FINAL**          | **{rec['score']:+.4f}**      | 100%  |                              |
    """)

    # Presentación visual de la recomendación
    color_bg = {"green": "#1a4d1a", "red": "#4d1a1a", "orange": "#4d3a00"}
    st.markdown(
        f"""
        <div style='
            background-color:{color_bg[rec["color"]]};
            border-left: 6px solid {rec["color"]};
            padding:20px; border-radius:8px; margin-top:10px;
        '>
            <h2 style='color:{rec["color"]}; margin:0;'>{rec["recomendacion"]}</h2>
            <p style='color:#ddd; margin:8px 0 0 0;'>{rec["descripcion"]}</p>
            <p style='color:#aaa; font-size:0.85rem; margin:4px 0 0 0;'>
                Score final ponderado: <strong>{rec['score']:+.4f}</strong> &nbsp;|&nbsp;
                Umbrales: <span style='color:green'>COMPRA > 0.2</span> &nbsp;|&nbsp;
                <span style='color:orange'>MANTENER ±0.2</span> &nbsp;|&nbsp;
                <span style='color:red'>VENTA &lt; -0.2</span>
            </p>
            <p style='color:#666; font-size:0.75rem; margin:8px 0 0 0;'>
                ⚠️ Este resultado es meramente informativo y no constituye asesoría financiera.
                Desarrollado por Diego CR.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PIE DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style='text-align:center; color:#555; font-size:0.8rem;'>
        Analizador de Portafolios & Valoración de Activos · Desarrollado por <strong>Diego CR</strong><br>
        Los resultados son meramente informativos y <strong>no constituyen asesoría de inversión</strong>.
        Datos obtenidos de Yahoo Finance vía <code>yfinance</code>.
    </div>
    """,
    unsafe_allow_html=True,
)
