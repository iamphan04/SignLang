@echo off
cd /d "%~dp0"
call D:\anaconda\Scripts\activate.bat signlang
python app_translation.py
pause
