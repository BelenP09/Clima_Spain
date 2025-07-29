# 🌾 Dashboard Climático Agrícola - Sevilla 

## 📋 Descripción

Dashboard interactivo profesional para el análisis climático de la estación meteorológica Sevilla con enfoque específico en aplicaciones agrícolas. 

Basado en el análisis exhaustivo realizado en `EDA.ipynb` e incorporando comparaciones con los estudios regionales de `EDA_Agricola.ipynb` y `EDA_Andalucia.ipynb`.

## 🎯 Características Principales

### 📊 Análisis Multidimensional
- **Temperaturas**: Evolución histórica con tendencias y proyecciones
- **Precipitación**: Patrones anuales y variabilidad climática
- **Eventos Extremos**: Días de calor intenso y heladas
- **Indicadores Agrícolas**: Grados día de crecimiento, estrés hídrico
- **Análisis ML**: Predicciones basadas en machine learning

### 🔧 Tecnología
- **Framework**: Streamlit para interfaces web interactivas
- **Visualización**: Plotly para gráficos dinámicos de alta calidad
- **Análisis**: pandas, numpy para procesamiento de datos
- **ML**: Basado en resultados del EDA con Random Forest

## 🚀 Instalación y Uso

### Requisitos Previos
- Python 3.8 o superior
- Acceso al archivo `Data/Base de datos/clima_limpio.csv`

### Instalación Rápida

#### Opción 1: Ejecutor Automático (Windows)
```bash
# Hacer doble clic en:
ejecutar_dashboard.bat
```

#### Opción 2: Manual
```bash
# 1. Instalar dependencias
pip install -r requirements_streamlit.txt

# 2. Ejecutar dashboard
streamlit run dashboard_climatico_real.py

# 3. Abrir navegador en http://localhost:8501
```

## 📊 Estructura del Dashboard

### 🌡️ Tab 1: Temperaturas
- Evolución histórica de temperaturas máximas y mínimas
- Tendencias lineales con significancia estadística
- Métricas de cambio climático
- Análisis de amplitud térmica

### 🌧️ Tab 2: Precipitación
- Patrones de precipitación anual
- Identificación de años secos/húmedos
- Tendencias a largo plazo
- Análisis de variabilidad

### ⚡ Tab 3: Eventos Extremos
- Días con temperaturas >35°C
- Frecuencia de heladas
- Tendencias de eventos extremos
- Evaluación de riesgo agrícola

### 🌾 Tab 4: Indicadores Agrícolas
- Grados Día de Crecimiento (GDD base 10°C)
- Días secos y estrés hídrico
- Evaluación de aptitud para cultivos
- Recomendaciones por tipo de cultivo

### 📊 Tab 5: Análisis Climático
- Clasificación climática por décadas
- Matriz de correlaciones entre variables
- Comparación con estudios regionales
- Estadísticas por período

### 🤖 Tab 6: ML & Predicciones
- Resultados de modelos de machine learning
- Proyecciones climáticas 2026-2050
- Escenarios de impacto agrícola
- Métricas de confianza del modelo

## 📈 Datos y Metodología

### Fuente de Datos
- **Estación**: SPE00120512 - Sevilla San Pablo
- **Período**: 1951-2025 (74 años de registros)
- **Variables**: Temperatura máx/mín, precipitación, eventos extremos
- **Calidad**: Filtrado por años con >300 días de datos

### Procesamiento
1. **Limpieza**: Conversión de unidades, filtrado de outliers
2. **Agregación**: Estadísticas anuales y por década
3. **Indicadores**: Cálculo de GDD, días extremos, índices de sequía
4. **ML**: Aplicación de Random Forest para predicciones

### Análisis Estadístico
- Tendencias lineales con test de significancia
- Correlaciones de Pearson entre variables
- Clasificación climática por anomalías de temperatura
- Proyecciones basadas en series temporales

## 🎨 Características de la Interfaz

### 🎛️ Controles Interactivos
- **Filtro temporal**: Selección de período de análisis
- **Métricas dinámicas**: Actualización automática según filtros
- **Navegación por tabs**: Organización temática del contenido

### 📊 Visualizaciones
- **Gráficos de tendencias**: Series temporales con líneas de regresión
- **Mapas de calor**: Matrices de correlación
- **Gráficos apilados**: Distribución de tipos climáticos
- **Proyecciones futuras**: Escenarios de cambio climático

### 📱 Responsive Design
- Adaptable a diferentes tamaños de pantalla
- Optimizado para presentaciones profesionales
- CSS personalizado para mejor legibilidad

## 🔬 Validación Científica

### Comparación con Literatura
- **Consistencia regional**: Alineado con patrones andaluces
- **Tendencias nacionales**: Coherente con estudios españoles
- **Indicadores agrícolas**: Validados con literatura agronómica

### Métricas de Calidad
- **R² Temperatura**: 0.603 (modelo Random Forest)
- **Significancia tendencias**: p < 0.001
- **Cobertura temporal**: 74 años de datos históricos

## 🌾 Aplicaciones Agrícolas

### Cultivos Recomendados
- **Excelente aptitud**: Vid, olivo, cítricos
- **Buena aptitud**: Cereales, leguminosas, hortalizas
- **Cultivos especializados**: Arroz (con riego), almendro

### Estrategias de Adaptación
1. **Eficiencia hídrica**: Riego por goteo, mulching
2. **Variedades resistentes**: Tolerancia a calor y sequía
3. **Calendario flexible**: Ajuste de fechas de siembra
4. **Infraestructura**: Sombreo, sistemas de refrigeración

## 📁 Estructura de Archivos

```
├── dashboard_climatico_real.py    # Dashboard principal con datos reales
├── dashboard_climatico.py         # Versión con datos sintéticos
├── requirements_streamlit.txt     # Dependencias de Python
├── ejecutar_dashboard.bat        # Script de ejecución automática
├── README_dashboard.md           # Esta documentación
└── Data/
    └── Base de datos/
        └── clima_limpio.csv      # Datos climáticos procesados
```

## 🔧 Personalización

### Modificar Umbrales
```python
# En el código, buscar y modificar:
EXTREME_HEAT_THRESHOLD = 35  # °C para días extremos
FROST_THRESHOLD = 0          # °C para heladas
GDD_BASE = 10               # °C base para grados día
```

### Agregar Nuevos Indicadores
```python
# Ejemplo: Índice de aridez
def calculate_aridity_index(temp, precip):
    return precip / (temp * 0.1)  # Simplificado
```

### Personalizar Visualizaciones
```python
# Cambiar colores, estilos en las funciones create_*_chart()
colors = {'Frío': '#your_color', 'Normal': '#your_color', 'Cálido': '#your_color'}
```

## 🚨 Solución de Problemas

### Error de Carga de Datos
```
Error: No se encontraron datos para la estación SPE00120512
```
**Solución**: Verificar que el archivo `clima_limpio.csv` existe y contiene la estación SPE00120512

### Error de Dependencias
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solución**: 
```bash
pip install -r requirements_streamlit.txt
```

### Puerto Ocupado
```
Error: Port 8501 is already in use
```
**Solución**:
```bash
streamlit run dashboard_climatico_real.py --server.port 8502
```

## 📞 Soporte y Contribución

### Reportar Problemas
- Crear issue en el repositorio
- Incluir mensaje de error completo
- Especificar sistema operativo y versión de Python

### Mejoras Sugeridas
- Nuevos indicadores agroclimáticos
- Visualizaciones adicionales
- Comparaciones con otras estaciones
- Integración con APIs meteorológicas

## 📄 Licencia

Este dashboard está basado en el análisis climático del proyecto Clima_Spain y sigue la misma licencia del proyecto principal.

---

**💡 Tip**: Para presentaciones profesionales, usar modo pantalla completa del navegador y ocultar la barra lateral de Streamlit.

**🔗 Enlaces útiles**:
- [Documentación Streamlit](https://docs.streamlit.io)
- [Plotly Python](https://plotly.com/python/)
- [AEMET - Datos Climáticos](https://www.aemet.es)
