@echo off
REM ============================================================
REM    CoreX OmniAnomaly - Full Local Run Script
REM    Run this file from the project root directory.
REM    It runs every stage in the correct order.
REM ============================================================

REM Fix Windows terminal encoding for emoji/unicode in print statements
set PYTHONIOENCODING=utf-8

REM Use the correct Python environment (Python 3.6 + TF 1.12)
set PYTHON=D:\miniconda3\envs\corex_env\python.exe

echo.
echo ============================================================
echo  STAGE 0: Check GPU
echo ============================================================
%PYTHON% check_gpu.py
echo.

echo ============================================================
echo  STAGE 1: Data Preprocessing + Feature Engineering
echo ============================================================
%PYTHON% data_preprocess.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Preprocessing failed. Fix errors above before continuing.
    pause
    exit /b 1
)
echo.

echo ============================================================
echo  STAGE 2: Inspect Processed Data (Optional Audit)
echo ============================================================
%PYTHON% inspect_pkl.py
echo.

echo ============================================================
echo  STAGE 3: Train the OmniAnomaly Model
echo    - Change --max_epoch 2 to --max_epoch 100 for full training
echo ============================================================
%PYTHON% main.py --max_epoch 100
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training failed. Fix errors above.
    pause
    exit /b 1
)
echo.

echo ============================================================
echo  STAGE 4: Plot Results and Save Report
echo ============================================================
%PYTHON% plot_results.py
echo.

echo ============================================================
echo  ALL STAGES COMPLETE!
echo  Check results\ folder and RobotArm_final_report.png
echo ============================================================
pause
