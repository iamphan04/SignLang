@echo off
chcp 65001 >nul
REM ===================================================
REM  Script chay webcam KSL - dung venv trong project
REM  (tranh loi AppControl cua D:\anaconda)
REM ===================================================

set PYTHON=d:\Project\SignLanguage\SignLang\.venv\Scripts\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

cd /d d:\Project\SignLanguage\SignLang\sign-language-recognition

echo Dang kiem tra cam 0 voi auto backend...
%PYTHON% 11_webcam.py --camera 0

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo Cam 0 that bai. Thu DirectShow backend...
    %PYTHON% 11_webcam.py --camera 0 --backend dshow
)

pause
