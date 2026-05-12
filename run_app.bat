@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

set NO_PROXY=127.0.0.1,localhost,*
set no_proxy=127.0.0.1,localhost,*
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=

cd /d "%ROOT%"
if errorlevel 1 (
    echo Cannot change to project directory
    pause
    exit /b 1
)

if not exist "%ROOT%\venv\Scripts\python.exe" (
    echo Error: venv not found. Run: python -m venv venv
    pause
    exit /b 1
)

if not exist "%ROOT%\app.py" (
    echo Error: app.py not found
    pause
    exit /b 1
)

"%ROOT%\venv\Scripts\python.exe" "%ROOT%\app.py"

pause
