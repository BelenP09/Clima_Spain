# 🌡️ Clima_Spain - Análisis Climático Integral de España

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Data Science](https://img.shields.io/badge/Data%20Science-Climate%20Analysis-orange.svg)

**Un proyecto completo de análisis exploratorio de datos climáticos de España con aplicaciones web interactivas**

[🚀 Demo España](#-demo-y-ejecución) • [🌡️ Demo Andalucía](#-análisis-específico-de-andalucía) • [📊 Notebooks](#-notebooks-de-análisis) • [📁 Estructura](#-estructura-del-proyecto)

</div>

---

## 📋 Tabla de Contenidos

- [🎯 Descripción del Proyecto](#-descripción-del-proyecto)
- [✨ Características Principales](#-características-principales)
- [🚀 Demo y Ejecución](#-demo-y-ejecución)
- [🌡️ Análisis Específico de Andalucía](#-análisis-específico-de-andalucía)
- [📊 Notebooks de Análisis](#-notebooks-de-análisis)
- [🛠️ Instalación](#️-instalación)
- [📁 Estructura del Proyecto](#-estructura-del-proyecto)
- [📈 Datos y Fuentes](#-datos-y-fuentes)
- [🔧 Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [📸 Capturas de Pantalla](#-capturas-de-pantalla)
- [🤝 Contribución](#-contribución)
- [📄 Licencia](#-licencia)

---

## 🎯 Descripción del Proyecto

**Clima_Spain** es un proyecto integral de análisis de datos climáticos que combina ciencia de datos, visualización interactiva y aplicaciones web para proporcionar insights profundos sobre el clima español. El proyecto incluye análisis temporales, geográficos, predictivos y específicos por regiones.

### 🔍 Objetivos

- **Análisis Temporal**: Evolución climática desde 1920 hasta 2025
- **Análisis Geográfico**: Distribución espacial de variables climáticas
- **Predicciones**: Proyecciones climáticas hasta 2055
- **Interactividad**: Dashboards web para exploración de datos
- **Regionalización**: Análisis específico de Andalucía

---

## ✨ Características Principales

### 🌍 **Análisis Nacional (España)**
- **📊 Resumen Ejecutivo**: Métricas clave y overview del dataset
- **📈 Análisis Temporal**: Evolución de temperaturas (1980-2025)
- **🗺️ Análisis Geográfico**: Distribución de +90 estaciones meteorológicas
- **🌡️ Variables Climáticas**: Temperatura máxima, mínima y precipitación
- **⚠️ Detección de Outliers**: Identificación automática de valores atípicos
- **🔮 Análisis Predictivo**: Modelos ML con proyecciones a 2050
- **🎯 Conclusiones**: Hallazgos clave y recomendaciones

### 🏛️ **Análisis Regional (Andalucía)**
- **📊 Estadísticas Descriptivas**: Métricas específicas de la región
- **🗺️ Mapas de Calor Interactivos**: Visualización mensual detallada
- **📈 Tendencias por Décadas**: Análisis histórico desde 1920
- **🔮 Predicciones Regionales**: Proyecciones específicas hasta 2055
- **📋 Resumen Ejecutivo**: Insights y recomendaciones regionales

### 🎨 **Visualizaciones Avanzadas**
- Gráficos interactivos con **Plotly**
- Mapas de calor dinámicos
- Series temporales con tendencias
- Boxplots para detección de outliers
- Matrices de correlación
- Proyecciones con múltiples escenarios

---

## 🚀 Demo y Ejecución

### **Método 1: Ejecución Automática (Recomendado)**
```bash
# Para análisis completo de España
run_streamlit.bat

# Para análisis específico de Andalucía
run_andalucia.bat
```

### **Método 2: Instalación Manual**
```bash
# 1. Clonar el repositorio
git clone https://github.com/BelenP09/Clima_Spain.git
cd Clima_Spain

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar aplicación principal
streamlit run streamlit_eda_app.py

# O ejecutar análisis de Andalucía
pip install -r requirements_andalucia.txt
streamlit run streamlit_eda_andalucia.py
```

### **Método 3: Entorno Virtual (Mejor Práctica)**
```bash
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar entorno (Windows)
venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar aplicación
streamlit run streamlit_eda_app.py
```

---

## 🌡️ Análisis Específico de Andalucía

### **🎯 Características Específicas**

#### **Tab 1: 📊 Estadísticas Generales**
- Métricas principales del clima andaluz
- Distribuciones de temperatura y precipitación
- Estadísticas por estación meteorológica
- Histogramas y box plots interactivos

#### **Tab 2: 🗺️ Mapas de Calor**
- Mapas interactivos mensuales
- Temperatura máxima, mínima y precipitación
- Análisis de patrones estacionales
- Insights automáticos por variable

#### **Tab 3: 📈 Tendencias Temporales**
- Análisis por décadas desde 1920
- Evolución temporal detallada
- Cálculo de cambios históricos
- Alertas sobre tendencias climáticas

#### **Tab 4: 🔮 Predicciones Climáticas**
- Modelos de regresión lineal
- Proyecciones hasta 2055
- Escenarios de cambio climático
- Recomendaciones estratégicas

---

## 📊 Notebooks de Análisis

### **1. 📓 EDA.ipynb** - Análisis Exploratorio Principal
- Limpieza y preprocesamiento de datos
- Análisis estadístico descriptivo
- Visualizaciones y correlaciones
- Detección de outliers y patrones

### **2. 🏛️ EDA_Andalucia.ipynb** - Análisis Regional
- Filtrado específico de datos andaluces
- Análisis temporal por décadas
- Mapas de calor regionales
- Predicciones específicas de la región

### **3. 🔧 preprocesamiento.ipynb** - Preparación de Datos
- Carga y unificación de archivos originales
- Limpieza y estandarización
- Creación de datasets procesados
- Validación de calidad de datos

---

## 🛠️ Instalación

### **Requisitos del Sistema**
- Python 3.8 o superior
- 4GB RAM mínimo (8GB recomendado)
- 2GB espacio en disco

### **Dependencias Principales**
```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.15.0
scikit-learn>=1.1.0
scipy>=1.9.0
```

### **Instalación Paso a Paso**
1. **Clonar repositorio**:
   ```bash
   git clone https://github.com/BelenP09/Clima_Spain.git
   cd Clima_Spain
   ```

2. **Crear entorno virtual**:
   ```bash
   python -m venv clima_env
   ```

3. **Activar entorno**:
   ```bash
   # Windows
   clima_env\Scripts\activate
   
   # Linux/Mac
   source clima_env/bin/activate
   ```

4. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Ejecutar aplicación**:
   ```bash
   streamlit run streamlit_eda_app.py
   ```

---

## 📁 Estructura del Proyecto

```
Clima_Spain/
├── 📊 Aplicaciones Web
│   ├── streamlit_eda_app.py          # App principal España
│   ├── streamlit_eda_andalucia.py    # App específica Andalucía
│   ├── run_streamlit.bat             # Launcher España
│   └── run_andalucia.bat             # Launcher Andalucía
│
├── 📓 Notebooks de Análisis
│   ├── EDA.ipynb                     # Análisis principal
│   ├── EDA_Andalucia.ipynb           # Análisis Andalucía
│   └── preprocesamiento.ipynb        # Preparación datos
│
├── 📋 Configuración
│   ├── requirements.txt              # Dependencias España
│   ├── requirements_andalucia.txt    # Dependencias Andalucía
│   ├── README.md                     # Documentación principal
│   ├── README_Andalucia.md           # Docs Andalucía
│   ├── README_Streamlit.md           # Docs aplicación web
│   └── LICENSE                       # Licencia MIT
│
└── 📂 Data/
    ├── clima.csv                     # Dataset principal
    ├── clima_limpio.csv              # Dataset procesado
    ├── columnas_por_archivo.csv      # Metadatos
    ├── Archivos originales/          # Datos fuente (90+ archivos)
    ├── PorEstacion/                  # Datos por estación
    ├── Base de datos/                # Estructuras DB
    └── SQL/                          # Scripts SQL
```

---

## 📈 Datos y Fuentes

### **📊 Dataset Principal**
- **Periodo**: 1920 - 2025 (105 años)
- **Estaciones**: 90+ estaciones meteorológicas
- **Variables**: Temperatura máx/mín, precipitación
- **Registros**: +2M observaciones
- **Cobertura**: Todo el territorio español

### **🌍 Variables Climáticas**
| Variable | Descripción | Unidad | Rango |
|----------|-------------|--------|--------|
| `TMAX` | Temperatura máxima diaria | °C | -15° a 50° |
| `TMIN` | Temperatura mínima diaria | °C | -25° a 35° |
| `PRCP` | Precipitación diaria | mm | 0 a 300+ |
| `DATE` | Fecha de observación | YYYY-MM-DD | 1920-2025 |
| `STATION` | Código estación | String | SP*/SPE*/SPM* |

### **📍 Cobertura Geográfica**
- **Andalucía**: 25+ estaciones
- **Cataluña**: 15+ estaciones
- **Madrid**: 10+ estaciones
- **Valencia**: 12+ estaciones
- **Otras regiones**: 30+ estaciones

---

## 🔧 Tecnologías Utilizadas

### **🐍 Backend y Análisis**
- **Python 3.8+**: Lenguaje principal
- **Pandas**: Manipulación de datos
- **NumPy**: Cálculos numéricos
- **SciPy**: Análisis estadístico
- **Scikit-learn**: Machine Learning

### **📊 Visualización**
- **Streamlit**: Framework web interactivo
- **Plotly**: Gráficos interactivos
- **Matplotlib**: Gráficos estáticos
- **Seaborn**: Visualizaciones estadísticas

### **🔧 Herramientas de Desarrollo**
- **Jupyter Notebooks**: Análisis exploratorio
- **Git**: Control de versiones
- **VS Code**: Editor principal
- **Batch Scripts**: Automatización Windows

---

## 📸 Capturas de Pantalla

### **🏠 Dashboard Principal**
*Resumen ejecutivo con métricas clave del clima español*

### **📈 Análisis Temporal**
*Evolución de temperaturas y tendencias climáticas*

### **🗺️ Análisis Geográfico**
*Distribución espacial de estaciones meteorológicas*

### **🔮 Predicciones Climáticas**
*Proyecciones futuras con modelos ML*

*(Las capturas se agregarán en próximas versiones)*

---

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Para contribuir:

### **🔧 Desarrollo**
1. Fork el proyecto
2. Crea una rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

### **📝 Reportar Issues**
- Usa las templates de issues
- Incluye información detallada
- Adjunta capturas si es visual

### **💡 Ideas y Sugerencias**
- Nuevas visualizaciones
- Análisis adicionales
- Mejoras de performance
- Nuevas regiones

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 📞 Contacto

**Desarrollado por**: BelenP09  
**Email**: [Incluir email si deseas]  
**GitHub**: [@BelenP09](https://github.com/BelenP09)  
**Proyecto**: [Clima_Spain](https://github.com/BelenP09/Clima_Spain)

---

<div align="center">

**⭐ Si este proyecto te ha sido útil, considera darle una estrella en GitHub ⭐**

*Última actualización: Julio 2025*

</div>
