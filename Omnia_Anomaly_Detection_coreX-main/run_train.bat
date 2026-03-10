@echo off
call conda activate corex_env
cd /d X:\Omnia_CoreX\Omnia_Anomaly_Detection_coreX-main
set PYTHONPATH=.
echo === Starting Training ===
python -u main.py
echo === Training Complete ===
pause
