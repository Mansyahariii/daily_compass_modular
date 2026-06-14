@echo off
title Daily Compass Dashboard
echo ===================================================
echo [🧭] Menyiapkan Aplikasi Daily Compass JST
echo ===================================================
echo.
echo 1. Memasang/Memeriksa Dependensi (requirements.txt)...
pip install -r requirements.txt
echo.
echo 2. Menjalankan Aplikasi Web Streamlit...
streamlit run app.py
echo.
pause
