@echo off
title Voice Biometric Setup
color 0A

echo ============================================
echo   Voice Biometric Authentication System
echo   One-Click Setup
echo ============================================
echo.

:: ── Check Python ──────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python found.

:: ── Check Node / npm ──────────────────────────
npm --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed.
    echo Please install Node.js from https://nodejs.org/
    echo After installing, close this window and run Setup.bat again.
    echo.
    start https://nodejs.org/
    pause
    exit /b 1
)
echo [OK] Node.js found.

:: ── Install resemblyzer ───────────────────────
echo.
echo Installing Python voice recognition package...
python -m pip install resemblyzer --prefer-binary
if errorlevel 1 (
    echo [ERROR] Failed to install resemblyzer.
    echo Try running this file as Administrator.
    pause
    exit /b 1
)
echo [OK] resemblyzer installed.

:: ── Install npm packages ──────────────────────
echo.
echo Installing Node packages...
call npm install
if errorlevel 1 (
    echo [ERROR] npm install failed.
    pause
    exit /b 1
)
echo [OK] Node packages installed.

:: ── Launch the app ────────────────────────────
echo.
echo ============================================
echo   Setup complete! Launching the app...
echo   Open http://127.0.0.1:8765 in your browser
echo   Default admin passcode: 5846
echo   Press Ctrl+C to stop the server
echo ============================================
echo.
call npm run local
pause
