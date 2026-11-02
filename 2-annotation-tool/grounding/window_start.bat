@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Grounding review UI (Windows)
REM Usage:
REM   double-click this file (grounding\window_start.bat)
REM   window_start.bat --port 8090
REM   window_start.bat --dataset D:\data\your_dataset
REM Close this window to stop the server and free THIS port.
REM Multi-instance: if PORT busy, auto use PORT+1 / +2 ... (keeps other windows).
REM
REM Python resolve order:
REM   1) <root>\runtime\python\python.exe        (portable zip)
REM   2) <root>\build\runtime\python\python.exe  (dev / pack runtime)
REM   3) conda env yulin
REM   4) python on PATH
REM Project root is this folder when util\ sits beside the bat,
REM otherwise the parent.
REM NOTE: keep this .bat ASCII-only. UTF-8 Chinese breaks cmd.exe on CN Windows.

cd /d "%~dp0"
if exist "%~dp0util\app.py" (
  for %%I in ("%~dp0.") do set "ROOT=%%~fI"
) else (
  for %%I in ("%~dp0..") do set "ROOT=%%~fI"
)
cd /d "%ROOT%"

REM ---- editable ----
REM Fallback when model\server.json has no default. UI can still switch services.
set "VLLM_BASE_URL=http://<ip>:8081/v1"
set "VLLM_MODEL=qwen3.8-27b"
set "DATASET="
set "PORT=8090"
set "PY_DEV=/path/to/python.exe"
REM ------------------

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

:parse_args
if "%~1"=="" goto after_args
if /i "%~1"=="--port" (
  set "PORT=%~2"
  shift
  shift
  goto parse_args
)
if /i "%~1"=="--dataset" (
  set "DATASET=%~2"
  shift
  shift
  goto parse_args
)
echo Unknown arg: %~1
shift
goto parse_args

:after_args

set "PY="
if exist "%ROOT%\runtime\python\python.exe" (
  set "PY=%ROOT%\runtime\python\python.exe"
) else if exist "%ROOT%\build\runtime\python\python.exe" (
  set "PY=%ROOT%\build\runtime\python\python.exe"
) else if exist "%PY_DEV%" (
  set "PY=%PY_DEV%"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] python not found.
    echo   Portable zip: runtime\python\
    echo   Dev tree:     build\runtime\python\
    echo   Or install Python / fix PY_DEV= in this bat.
    pause
    exit /b 1
  )
  set "PY=python"
)

"%PY%" -c "import flask" 2>nul
if errorlevel 1 (
  echo Installing: flask pillow tqdm requests ...
  "%PY%" -m pip install flask pillow tqdm requests -q
  if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
  )
)

echo.
echo   Grounding review UI
echo   preferred port: %PORT%  (auto +1 if busy)
if defined DATASET (
  echo   dataset: %DATASET%
) else (
  echo   dataset: not set, pick in UI
)
echo   vLLM: %VLLM_BASE_URL%
echo   model: %VLLM_MODEL%
echo   python: %PY%
echo   Close this window to free THIS instance port
echo.

if not exist "%ROOT%\util\win_job_run.ps1" (
  echo [ERROR] missing util\win_job_run.ps1 under %ROOT%
  pause
  exit /b 1
)

REM Job Object: closing console kills python child; port auto-increments if busy
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\util\win_job_run.ps1" -Port %PORT% -Dataset "%DATASET%" -Python "%PY%" -OpenBrowser
set "ERR=%ERRORLEVEL%"

echo.
if not "%ERR%"=="0" (
  echo Service exited, code %ERR%
) else (
  echo Service exited, this instance port freed
)
pause
endlocal & exit /b %ERR%
