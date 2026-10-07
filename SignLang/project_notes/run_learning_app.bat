@echo off
chcp 65001 >nul
REM ===================================================
REM  Streamlit Learning App Launcher
REM ===================================================

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

cd /d d:\Project\SignLanguage\SignLang\sign-language-recognition
echo Dang mo ung dung Streamlit Hoc VSL...
echo.
d:\Project\SignLanguage\SignLang\.venv\Scripts\streamlit.exe run app_learning.py
pause
