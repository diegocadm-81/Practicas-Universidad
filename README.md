# 📈 Analizador de Portafolios & Valoración de Activos
**Autor: Diego CR**

---

## Descripción

Aplicación Streamlit en Python para análisis completo de portafolios de inversión y valoración de activos individuales. Los datos se descargan automáticamente desde **Yahoo Finance** vía `yfinance`.

---

## Funcionalidades

### Módulo 1 – Análisis de Portafolio
- Descarga automática de precios ajustados (Yahoo Finance)
- Gráfica de precios en **base 1000** (compara activos de distintas magnitudes)
- Gráfica de **rendimientos históricos acumulados**
- Tabla de **métricas anualizadas**: Rendimiento EA, Volatilidad EA, Sharpe Ratio
- **Métricas de riesgo**: Beta, VaR 95%, CVaR 95% respecto al benchmark
- **Matriz de correlación** interactiva (heatmap)
- **Comparación vs benchmark** (portafolio igual peso vs. índice de referencia)

### Módulo 2 – Optimización de Portafolio
Tres métodos con tabla comparativa de pesos y métricas:
1. **Markowitz** – Mínima varianza (eficiencia sin cortos)
2. **CAPM / Máx Sharpe** – Portafolio de tangencia (máximo rendimiento por unidad de riesgo)
3. **Monte Carlo** – 5,000 portafolios aleatorios, selecciona el de mayor Sharpe

### Módulo 3 – Simulación Monte Carlo de Precios
- **1,000 trayectorias** a 252 días usando movimiento browniano geométrico (GBM)
- Percentiles P5, P50, P95 del precio simulado
- Histograma de distribución de precios al final del horizonte

### Módulo 4 – Valoración de Ticker Individual
Valora cualquier ticker del portafolio con tres enfoques:

#### A. Análisis Técnico
- Medias Móviles: MA5, MA10, MA20, MA200 (Golden/Death Cross)
- RSI (14 días) – zonas sobrecompra/sobreventa
- MACD y Signal (12-26-9)
- Niveles de Fibonacci (23.6%, 38.2%, 50%, 61.8%, 100%)

#### B. Análisis Estadístico
- **Regresión lineal** ln(ticker) ~ ln(benchmark): precio objetivo y potencial
- **Percentiles históricos**: en qué nivel histórico está el precio actual

#### C. Análisis Fundamental
- Múltiplos de mercado: P/E, P/B, EV/EBITDA, ROE, Yield
- **DCF (Flujo de Caja Libre Descontado)**: formulario editable con WACC, FCF, valor terminal
- Score fundamental basado en potencial del modelo

#### Recomendación Final Ponderada
- El usuario define el peso de cada tipo de valoración (Técnico / Estadístico / Fundamental)
- Score final entre -1 y +1
  - `> 0.2` → **🟢 COMPRA** (infravalorada)
  - `−0.2 a 0.2` → **🟡 MANTENER** (precio justo)
  - `< −0.2` → **🔴 VENTA** (sobrevalorada)

---

## Instalación y ejecución

### 1. Clonar o descargar el archivo
Guarda `app_portafolio.py` y `requirements.txt` en la misma carpeta.

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación
```bash
streamlit run app_portafolio.py
```

La app se abrirá automáticamente en tu navegador en `http://localhost:8501`.

---

## Cómo usar la app

1. En el **panel lateral izquierdo**:
   - Ingresa los tickers separados por coma (ej: `SPY, QQQ, EEM, GLD`)
   - Ingresa el ticker del benchmark (ej: `SPY`)
   - Selecciona el período de análisis (1, 3 o 5 años)
   - Ajusta la tasa libre de riesgo si lo deseas

2. Haz clic en **"🚀 Analizar Portafolio"**

3. Navega por las 4 pestañas:
   - **📊 Análisis de Portafolio** – métricas y gráficas históricas
   - **⚙️ Optimización** – los tres portafolios óptimos
   - **🎲 Monte Carlo** – simulación de precios futuros
   - **🔍 Valoración Ticker** – análisis técnico, estadístico y fundamental

---

## Notas importantes

- Los datos son descargados en tiempo real desde Yahoo Finance
- Los resultados son **meramente informativos** y no constituyen asesoría financiera
- Para la valoración fundamental por DCF, los datos financieros deben ingresarse manualmente
- Para activos latinoamericanos o locales que no estén en Yahoo Finance, se recomienda usar sus ADRs (ej: `EC` para Ecopetrol, `CIB` para Bancolombia)

---

*Desarrollado por **Diego CR** · Maestría en Administración Financiera*
