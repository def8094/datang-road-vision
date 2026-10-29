@echo off
rem 双击启动大唐道路检测界面（使用项目自带的 .venv 环境）
cd /d "%~dp0"
".venv\Scripts\python.exe" app\main.py
