@echo off
chcp 65001 >nul
echo Запуск приложения "ИИ-сценарист Дальнего Востока"...
echo.

REM Переход в папку, где лежит этот .bat файл (и проект)
cd /d "%~dp0"

REM Активация виртуального окружения (если оно есть)
REM Если вы не используете venv, закомментируйте эту строку знаком REM
REM call venv\Scripts\activate

REM Запуск Streamlit
streamlit run app.py --server.port 8501

pause