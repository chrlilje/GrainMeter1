@echo off
pushd "%~dp0"
REM Run without console if pythonw is available, otherwise use python
if exist ".venv\Scripts\pythonw.exe" (
    .venv\Scripts\pythonw.exe .\grain_measure_app\main.py
) else (
    .venv\Scripts\python.exe .\grain_measure_app\main.py
)
popd
