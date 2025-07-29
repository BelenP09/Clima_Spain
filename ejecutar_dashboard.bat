@echo off
echo 🌾 Iniciando Dashboard Climático Agrícola - Sevilla 
echo.
echo 📋 Verificando dependencias...

REM Verificar si Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python no está instalado o no está en el PATH
    echo 📥 Por favor, instala Python desde https://python.org
    pause
    exit /b 1
)

REM Verificar si Streamlit está instalado
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📦 Instalando dependencias...
    pip install -r requirements_streamlit.txt
    if errorlevel 1 (
        echo ❌ Error al instalar dependencias
        pause
        exit /b 1
    )
)

echo ✅ Dependencias verificadas
echo.
echo 🚀 Iniciando dashboard en el navegador...
echo 📍 URL: http://localhost:8501
echo.
echo ⚠️  Para detener el servidor: Ctrl+C en esta ventana
echo.

REM Ejecutar Streamlit
streamlit run dashboard_climatico_real.py --server.port 8501 --server.address localhost

pause
