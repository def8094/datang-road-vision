#!/usr/bin/env python3
"""Regenerate the desktop VBS launcher in GBK (ANSI) encoding for WScript."""
from pathlib import Path

VBS = Path(r"C:\Users\Administrator\Desktop\大唐道路检测.vbs")
PROJ = r"C:\Users\Administrator\WorkBuddy\大唐"
PYW = PROJ + r"\.venv\Scripts\pythonw.exe"
MAIN = PROJ + r"\app\main.py"

lines = [
    "' 大唐道路检测界面 - 双击启动（无控制台窗口）",
    'Set ws = CreateObject("WScript.Shell")',
    f'ws.CurrentDirectory = "{PROJ}"',
    f'ws.Run """{PYW}"" ""{MAIN}""", 0, False',
]
VBS.write_text("\r\n".join(lines) + "\r\n", encoding="gbk")
print("written:", VBS, VBS.stat().st_size, "bytes (GBK)")
