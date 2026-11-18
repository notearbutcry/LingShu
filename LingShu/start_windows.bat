@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
python check_environment.py
if errorlevel 1 (
  echo 环境检查未通过。请先在 Anaconda Prompt 激活 Python 3.12 环境并安装依赖。
  pause
  exit /b 1
)
python -m streamlit run app.py
pause
