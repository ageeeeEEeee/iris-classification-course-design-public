@echo off
setlocal
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0photo\identify.py" %*
) else if exist "%~dp0..\iris-env\Scripts\python.exe" (
    "%~dp0..\iris-env\Scripts\python.exe" "%~dp0photo\identify.py" %*
) else (
    py -3.13 "%~dp0photo\identify.py" %*
)
if errorlevel 1 pause
