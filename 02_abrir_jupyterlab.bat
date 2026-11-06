@echo off
setlocal
cd /d "%~dp0"
call conda activate education-dynamics
jupyter lab
