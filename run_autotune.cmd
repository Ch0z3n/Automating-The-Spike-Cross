@echo off
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src"
echo The Spike Cross autotuner
echo Keep the game visible and focused. Press F12 in the game to stop.
echo.
".venv\Scripts\python.exe" -m spike_bot --config config.json --autotune
echo.
echo Autotuner stopped. Results are in autotune_state.json and logs\
pause
