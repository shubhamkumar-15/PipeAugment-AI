@echo off
title PipeAugment AI - Development Launcher
color 0E
echo =====================================================================
echo                PIPEAUGMENT AI — ONE-CLICK LAUNCHER
echo =====================================================================
echo.
echo Launching Backend API, AI GPU Worker, and Next.js Frontend...
echo Your web browser will open automatically once the services start.
echo.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-dev.ps1"
pause
