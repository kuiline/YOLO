@echo off
title 植物病害监测系统 - 全自动控制台
color 0A

echo ======================================================
echo    正在启动植物病害监测系统 (开发监听模式)
echo ======================================================

:: 1. 检查虚拟环境
if not exist "venv\Scripts\python.exe" (
    echo [错误] 找不到 venv 虚拟环境，请确保它在当前文件夹下。
    pause
    exit
)

:: 2. 异步启动浏览器
:: 利用 start 命令的异步特性，等待 5 秒后自动打开网页
echo [启动] 正在预热浏览器，请稍候...
start /b cmd /c "timeout /t 6 >nul && start http://127.0.0.1:7860"

:: 3. 启动 Gradio 热重载模式
echo [提示] 只要修改 .py 代码并保存，系统就会自动重启
echo ======================================================
.\venv\Scripts\python.exe -m gradio app.py

pause
