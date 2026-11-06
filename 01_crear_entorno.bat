@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo CREACION DEL ENTORNO education-dynamics
echo ================================================
call conda env create -f environment.yml
if errorlevel 1 (
  echo.
  echo ERROR: No se pudo crear el entorno.
  echo Copia el mensaje completo y envialo para revisarlo.
  pause
  exit /b 1
)
call conda activate education-dynamics
python -m ipykernel install --user --name education-dynamics --display-name "Python (education-dynamics)"
conda env export --from-history > environment_history.yml
python scripts\verificar_entorno.py
if errorlevel 1 (
  echo ERROR: La verificacion fallo.
  pause
  exit /b 1
)
echo.
echo ENTORNO CREADO Y VERIFICADO CORRECTAMENTE.
pause
