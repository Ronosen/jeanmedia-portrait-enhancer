@echo off
setlocal
if not exist "%~dp0APP-PATH.txt" goto source
set /p JEANMEDIA_APP_REL=<"%~dp0APP-PATH.txt"
if not exist "%~dp0%JEANMEDIA_APP_REL%" goto source
set "APP=%~dp0%JEANMEDIA_APP_REL%"
for %%I in ("%APP%") do set "JEANMEDIA_APP_DIR=%%~dpI"
set "JEANMEDIA_PYTHON=%JEANMEDIA_APP_DIR%python\python.exe"
set "JEANMEDIA_WORKER="
set "PYTHONPATH=%JEANMEDIA_APP_DIR%.worker-deps"
set "PYTHONUTF8=1"
goto launch
:source
set "APP=%~dp0src-tauri\target\debug\jeanmedia.exe"
if not exist "%APP%" (
    echo Build JeanMedia first. See README.md.
    pause
    exit /b 1
)
if not defined JEANMEDIA_PYTHON if exist "%~dp0.venv\Scripts\python.exe" set "JEANMEDIA_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined JEANMEDIA_PYTHON if exist "%~dp0.worker-deps\cv2" (
    set "JEANMEDIA_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    set "PYTHONPATH=%~dp0.worker-deps"
)
:launch
start "" "%APP%"
