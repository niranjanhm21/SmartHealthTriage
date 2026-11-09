@echo off
echo Starting SmartHealth Triage...
cd /d "%~dp0.."
python -m streamlit run app/app.py
pause
