@echo off
REM Set encoding to UTF-8 to prevent emoji print errors on Windows Console
set PYTHONIOENCODING=utf-8
echo --- Running Data Preprocessor ---
python data_preprocess.py
pause
