# 🌾 Clima Spain - Análisis Climático Agrícola 

<div align="center">

![Climate Analysis](https://img.shields.io/badge/Climate-Analysis-green?style=for-the-badge&logo=leaflet)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**🚀 Plataforma avanzada de análisis climático para la agricultura mediterránea**

*Transformando datos meteorológicos en inteligencia agrícola para el futuro sostenible*

[📊 **Dashboard en Vivo**](http://localhost:8505) • [📚 **Documentación**](#documentación) • [🛠️ **Instalación**](#instalación) • [🌟 **Características**](#características)

</div>

---

## 🌟 Resumen del Proyecto

**Clima Spain** es una plataforma integral de análisis climático diseñada específicamente para optimizar la agricultura mediterránea. Combina **ciencia de datos**, **machine learning** y **visualización interactiva** para transformar décadas de datos meteorológicos en insights accionables para agricultores, investigadores y tomadores de decisiones.

### 🎯 **Misión**
Democratizar el acceso a análisis climáticos avanzados para impulsar una agricultura más resiliente, productiva y sostenible en España.

### 🔬 **Alcance Científico**
- **75+ años** de datos climáticos históricos (1951-2025)
- **8 provincias** de Andalucía con cobertura completa
- **+100 estaciones** meteorológicas analizadas
- **Machine Learning** para predicciones y clasificación climática

---

## ⚡ Características Principales

### 📈 **Dashboard Interactivo en Tiempo Real**
- **🗺️ Mapa de Estaciones**: Red completa de estaciones meteorológicas de Andalucía
- **🌡️ Análisis de Temperaturas**: Tendencias, anomalías y proyecciones futuras
- **🌧️ Patrones de Precipitación**: Variabilidad interanual y análisis de sequías
- **⚡ Eventos Extremos**: Monitoreo de olas de calor, heladas y días secos
- **🌾 Indicadores Agroclimáticos**: GDD, estrés hídrico y aptitud de cultivos
- **🤖 Machine Learning**: Predicciones y clasificación climática automatizada

### 🔍 **Análisis Exploratorio Avanzado**
- **EDA Sevilla**: Análisis detallado de la estación principal
- **EDA Andalucía**: Comparativa regional y patrones espaciales
- **EDA Agrícola**: Indicadores especializados para agricultura mediterránea
- **Preprocesamiento**: Pipeline automatizado de limpieza y validación de datos

### 📊 **Visualización Científica**
- Gráficos interactivos con **Plotly**
- Mapas georreferenciados de alta precisión
- Matrices de correlación y análisis multivariable
- Proyecciones climáticas hasta 2050

---

## 🛠️ Instalación

### 📋 **Requisitos del Sistema**
- **Python 3.11+**
- **Windows/Linux/macOS**
- **4GB RAM** mínimo
- **Conexión a internet** para mapas

### 🚀 **Instalación Rápida**

```bash
# 1. Clonar el repositorio
git clone https://github.com/BelenP09/Clima_Spain.git
cd Clima_Spain

# 2. Crear entorno virtual (recomendado)
python -m venv clima_env
source clima_env/bin/activate  # Linux/Mac
# clima_env\Scripts\activate    # Windows

# 3. Instalar dependencias
pip install -r requirements_streamlit.txt

# 4. Ejecutar dashboard
streamlit run dashboard_climatico_real.py
```

### 🎯 **Instalación con Ejecutable (Windows)**
```cmd
# Ejecutar directamente el archivo batch
ejecutar_dashboard.bat
```

### 📦 **Dependencias Principales**
```python
streamlit==1.28.1    # Framework web interactivo
pandas==2.0.3        # Manipulación de datos
numpy==1.24.3        # Computación científica
plotly==5.15.0       # Visualización interactiva
scipy==1.11.1        # Análisis estadístico
```

---

## 🏗️ Arquitectura del Proyecto

```
📁 Clima_Spain/
├── 🎯 dashboard_climatico_real.py    # Dashboard principal de Streamlit
├── 🔧 ejecutar_dashboard.bat         # Script de ejecución rápida
├── 📋 requirements_streamlit.txt     # Dependencias del proyecto
├── 📊 Data/                          # Almacén de datos climáticos
│   ├── 📂 Archivos originales/       # Datos sin procesar (CSV)
│   ├── 🗄️ Base de datos/            # Datos procesados y limpios
│   ├── 🏭 PorEstacion/              # Datos organizados por estación
│   └── 💾 SQL/                      # Scripts y consultas SQL
├── 📓 Notebook/                      # Análisis exploratorio (Jupyter)
│   ├── 🔬 EDA_Sevilla.ipynb         # Análisis estación principal
│   ├── 🌍 EDA_Andalucia.ipynb       # Análisis regional
│   ├── 🌾 EDA_Agricola.ipynb        # Indicadores agroclimáticos
│   ├── 🛠️ preprocesamiento.ipynb    # Pipeline de datos
│   └── 📈 Resultados_EDA_Agricola/  # Resultados y visualizaciones
└── 📚 README.md                     # Documentación del proyecto
```

---

## 🎮 Guía de Uso

### 🚀 **Inicio Rápido**

1. **Ejecutar el Dashboard**
   ```bash
   streamlit run dashboard_climatico_real.py
   ```

2. **Acceder a la Interfaz**
   - Abrir navegador en: `http://localhost:8501`
   - El dashboard se carga automáticamente

3. **Navegación por Pestañas**
   - **🗺️ Mapa Estaciones**: Explorar la red meteorológica
   - **🌡️ Temperaturas**: Analizar tendencias térmicas
   - **🌧️ Precipitación**: Estudiar patrones de lluvia
   - **⚡ Eventos Extremos**: Monitorear extremos climáticos
   - **🌾 Indicadores Agrícolas**: Evaluar aptitud de cultivos
   - **📊 Análisis Climático**: Correlaciones y tendencias
   - **🤖 ML & Predicciones**: Modelos predictivos

### 🎛️ **Controles Interactivos**

- **📅 Filtro Temporal**: Seleccionar período de análisis
- **📊 Métricas Dinámicas**: Visualizar cambios en tiempo real
- **🗺️ Zoom en Mapas**: Explorar estaciones específicas
- **📈 Hover Interactivo**: Detalles al pasar el mouse

---

## 📊 Resultados y Hallazgos Clave

### 🌡️ **Tendencias de Temperatura**
- **📈 Calentamiento**: +0.186°C por década (1951-2025)
- **🔥 Eventos Extremos**: Incremento de 3 días >35°C por década
- **❄️ Heladas**: Reducción de 0.2 días <0°C por década

### 🌧️ **Patrones de Precipitación**
- **📉 Tendencia**: -28.2 mm por década
- **🌊 Variabilidad**: Alta fluctuación interanual (CV > 0.3)
- **🏜️ Sequías**: Incremento en frecuencia e intensidad

### 🌾 **Impacto Agrícola**
- **✅ GDD Favorables**: 1,800+ grados-día anuales
- **⚠️ Estrés Hídrico**: 81% de días secos anuales
- **🍇 Cultivos Ideales**: Vid, olivo, cítricos, hortalizas

### 🤖 **Machine Learning**
- **🎯 Temperatura**: R² = 0.603 (predicción confiable)
- **🌧️ Precipitación**: R² = -0.197 (alta variabilidad natural)
- **📊 Clasificación**: 87% precisión en tipos climáticos

---

## 🎯 Casos de Uso

### 👨‍🌾 **Para Agricultores**
- **📅 Planificación de Siembras**: Optimizar calendario agrícola
- **💧 Gestión del Riego**: Anticipar períodos secos
- **🌱 Selección de Variedades**: Elegir cultivos resistentes
- **🛡️ Gestión de Riesgos**: Prepararse para eventos extremos

### 🔬 **Para Investigadores**
- **📈 Análisis de Tendencias**: Estudiar cambio climático regional
- **🧮 Modelado Climático**: Validar modelos con datos históricos
- **📊 Publicaciones**: Generar gráficos y estadísticas científicas
- **🌍 Comparaciones**: Benchmarking con otras regiones

### 🏛️ **Para Tomadores de Decisiones**
- **📋 Políticas Agrícolas**: Diseñar estrategias de adaptación
- **💰 Seguros Agrícolas**: Evaluar riesgos climáticos
- **🏗️ Infraestructura**: Planificar obras hidráulicas
- **📊 Reportes**: Generar informes para stakeholders

---

## 🧠 Metodología Científica

### 📊 **Fuentes de Datos**
- **🏢 NOAA Global Historical Climatology Network**: Datos validados internacionalmente
- **📍 Red de Estaciones**: +100 estaciones meteorológicas de Andalucía
- **📅 Cobertura Temporal**: 75 años de registros (1951-2025)
- **🔍 Variables**: Temperatura máx/mín, precipitación, coordenadas, elevación

### 🛠️ **Pipeline de Procesamiento**
1. **📥 Extracción**: Lectura de archivos CSV por chunks
2. **🧹 Limpieza**: Validación y conversión de unidades
3. **🔄 Transformación**: Cálculo de variables derivadas
4. **✅ Validación**: Control de calidad y detección de outliers
5. **📊 Agregación**: Resúmenes anuales y por estación

### 🤖 **Modelos de Machine Learning**
- **🌡️ Random Forest**: Predicción de temperaturas
- **🎯 K-Means Clustering**: Clasificación climática
- **📈 Regresión Lineal**: Análisis de tendencias
- **🔮 Proyección**: Extrapolación hasta 2050

### 📏 **Métricas de Evaluación**
- **R² Score**: Bondad de ajuste de modelos
- **RMSE**: Error cuadrático medio
- **MAE**: Error absoluto medio
- **Precisión**: Accuracy en clasificación

---

## 🌍 Impacto y Beneficios

### 🌱 **Sostenibilidad Ambiental**
- **💧 Eficiencia Hídrica**: Optimización del uso del agua
- **🌿 Agricultura Regenerativa**: Prácticas sostenibles basadas en datos
- **🔄 Adaptación Climática**: Estrategias de resiliencia agrícola
- **📉 Reducción de Emisiones**: Agricultura de precisión

### 💰 **Beneficios Económicos**
- **📈 Productividad**: Incremento del rendimiento de cultivos
- **💸 Reducción de Costes**: Optimización de recursos
- **🛡️ Gestión de Riesgos**: Menor impacto de eventos extremos
- **📊 Decisiones Informadas**: ROI mejorado en inversiones

### 🎓 **Valor Educativo**
- **📚 Democratización**: Acceso libre a análisis climáticos
- **🔬 Metodología Abierta**: Reproducibilidad científica
- **💡 Capacitación**: Herramienta educativa para profesionales
- **🌐 Transferencia de Conocimiento**: Aplicable a otras regiones

---

## 🔮 Roadmap y Futuro

### 🚀 **Versión 2.0** (Q3 2025)
- [ ] **🌍 Expansión Nacional**: Cobertura de toda España
- [ ] **📱 Aplicación Móvil**: Dashboard responsive para dispositivos móviles
- [ ] **🤖 IA Avanzada**: Deep Learning para predicciones más precisas
- [ ] **🔔 Alertas**: Sistema de notificaciones automáticas

### 🛠️ **Versión 2.5** (Q4 2025)
- [ ] **🛰️ Datos Satelitales**: Integración con imágenes de satélite
- [ ] **📊 Dashboard Personalizable**: Widgets configurables por usuario
- [ ] **🔌 API RESTful**: Acceso programático a datos y modelos
- [ ] **🌐 Multiidioma**: Soporte para inglés y francés

### 🌟 **Versión 3.0** (Q1 2026)
- [ ] **☁️ Cloud Native**: Migración a infraestructura en la nube
- [ ] **🤝 Colaboración**: Funciones multiusuario y compartir análisis
- [ ] **🧬 Modelos Específicos**: IA especializada por tipo de cultivo
- [ ] **📈 Mercados**: Integración con precios de commodities agrícolas

---

## 👥 Contribución

### 🤝 **¿Cómo Contribuir?**

1. **🍴 Fork** el repositorio
2. **🌿 Crear** una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. **💻 Desarrollar** tu contribución
4. **✅ Commit** tus cambios (`git commit -m 'Añadir nueva funcionalidad'`)
5. **📤 Push** a la rama (`git push origin feature/nueva-funcionalidad`)
6. **🔄 Abrir** un Pull Request

### 📋 **Tipos de Contribución**
- **🐛 Bug Reports**: Reportar errores o problemas
- **💡 Feature Requests**: Proponer nuevas funcionalidades
- **📚 Documentación**: Mejorar README, comentarios o tutoriales
- **🔧 Código**: Implementar features, optimizaciones o correcciones
- **🎨 Diseño**: Mejorar UX/UI del dashboard
- **📊 Datos**: Contribuir con nuevas fuentes de datos

### 🌟 **Reconocimientos**
Todos los contribuidores serán reconocidos en nuestra [página de contribuidores](CONTRIBUTORS.md) y en el dashboard.

---

## 📝 Licencia

Este proyecto está licenciado bajo la **MIT License** - ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License - Copyright (c) 2025 Belén

✅ Uso comercial permitido
✅ Modificación permitida  
✅ Distribución permitida
✅ Uso privado permitido
❌ Sin garantías
❌ Sin responsabilidad del autor
```

---

## 📞 Contacto y Soporte

### 👩‍💻 **Autora Principal**
**Belén** - *Climate Data Scientist & Agricultural Intelligence Developer*

### 📧 **Canales de Comunicación**
- **📊 GitHub Issues**: [Reportar problemas](https://github.com/BelenP09/Clima_Spain/issues)
- **💬 Discussions**: [Preguntas y sugerencias](https://github.com/BelenP09/Clima_Spain/discussions)
- **🐛 Bug Reports**: Usar template de issues para reportes detallados

### 📚 **Recursos Adicionales**
- **📖 Wiki**: [Documentación extendida](https://github.com/BelenP09/Clima_Spain/wiki)
- **🎥 Tutoriales**: [Videos explicativos](docs/tutorials/)
- **📊 Ejemplos**: [Casos de uso detallados](examples/)

---

## 🏆 Reconocimientos

### 🙏 **Agradecimientos**
- **🏢 NOAA**: Por proporcionar datos climáticos de alta calidad
- **🐍 Python Community**: Por las excelentes librerías de ciencia de datos
- **🌐 Streamlit**: Por democratizar el desarrollo de aplicaciones web
- **📊 Plotly**: Por las herramientas de visualización interactiva

### 🏅 **Inspiración**
Este proyecto está inspirado en la necesidad de democratizar el acceso a análisis climáticos avanzados para impulsar una agricultura más sostenible y resiliente ante el cambio climático.

---

<div align="center">

### 🌟 **¡Dale una estrella si este proyecto te ha sido útil!** ⭐

**Desarrollado con ❤️ para la comunidad agrícola española**

![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-💚-green.svg)
![Climate Science](https://img.shields.io/badge/Climate%20Science-🌍-blue.svg)

---

*"La mejor manera de predecir el futuro es crearlo basándose en datos del pasado"*

</div>