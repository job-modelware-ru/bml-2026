@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

where python >nul 2>nul
if errorlevel 1 (
    echo [!] Python not found. Install Python 3.9+ from python.org and tick "Add to PATH".
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment .venv ...
    python -m venv .venv || goto :fail
)
set PY=.venv\Scripts\python.exe

echo [2/4] Installing packages ...
"%PY%" -m pip install --upgrade pip -q
"%PY%" -m pip install -r requirements.txt -q || goto :fail

if not exist results mkdir results
echo [3/4] Running examples 01-06 (output -> output\) ...
"%PY%" run_all.py > results\examples_log.txt 2>&1 || goto :fail
type results\examples_log.txt

echo [4/4] Running benchmarks, about 15-25 minutes. Do not use the PC heavily meanwhile ...
"%PY%" bench_all.py || goto :fail

echo.
echo Done. Results are in the "results" and "output" folders.
pause
exit /b 0

:fail
echo.
echo [!] Something went wrong - see the messages above.
pause
exit /b 1
