@echo off
cd /d "%~dp0.."
uv run pytest -q
if errorlevel 1 exit /b 1
uv run python scripts\integration.py
if errorlevel 1 exit /b 1
uv run python scripts\recovery_checks.py
pause
