@echo off
title Rental Car BI Dashboard
cd /d "%~dp0"
if not exist rental_dw.db python build_warehouse.py
python -m streamlit run app.py
pause
