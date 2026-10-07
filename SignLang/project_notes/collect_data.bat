@echo off
chcp 65001 >nul
echo =====================================================
echo   VSL Data Collector - Thu thap du lieu ky hieu tay
echo =====================================================
echo.
echo Nhap ID nguoi thu thap (vd: p1, p2, p3):
set /p PERSON_ID=Person ID: 
if "%PERSON_ID%"=="" set PERSON_ID=p1

echo.
echo Nhap so mau muc tieu moi ky hieu (Enter = 250):
set /p TARGET=Target (mac dinh 250): 
if "%TARGET%"=="" set TARGET=250

echo.
echo Dang khoi dong... Person=%PERSON_ID%, Target=%TARGET%
echo.

cd /d d:\Project\SignLanguage\SignLang\sign-language-recognition

REM Dung Python tu venv trong project (tranh loi AppControl cua D:\anaconda)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHON=d:\Project\SignLanguage\SignLang\.venv\Scripts\python.exe

%PYTHON% collect_vsl.py --person %PERSON_ID% --target %TARGET% --camera 0

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo Thu voi backend DirectShow...
    %PYTHON% collect_vsl.py --person %PERSON_ID% --target %TARGET% --camera 0 --backend dshow
)

echo.
echo Hoan thanh! File CSV luu tai:
echo   data\vsl_collected\vsl_%PERSON_ID%.csv
echo.
pause
