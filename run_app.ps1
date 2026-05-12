# 植物病害检测系统 - 全自动启动脚本
# 功能：自动热重载代码 + 自动打开浏览器

$ServerUrl = "http://127.0.0.1:7860"

# 1. 确保环境正常
if (!(Test-Path "venv")) {
    Write-Host "--- [错误] 未发现虚拟环境 venv，请先创建它 ---" -ForegroundColor Red
    exit
}

Write-Host "--- [启动] 正在以热重载模式启动 Gradio 终端... ---" -ForegroundColor Cyan
Write-Host "--- [提示] 修改代码后保存，系统将自动重新加载 ---" -ForegroundColor Gray

# 2. 异步启动浏览器（等待端口开放）
Start-Job -ScriptBlock {
    $url = $using:ServerUrl
    while ($true) {
        try {
            $tcp = New-Object System.Net.Sockets.TcpClient
            $tcp.Connect("127.0.0.1", 7860)
            if ($tcp.Connected) {
                $tcp.Close()
                Write-Host "--- [检测] 端口已开放，正在打开浏览器... ---"
                Start-Process $url
                break
            }
        } catch {
            # 继续等待
        }
        Start-Sleep -Seconds 1
    }
} | Out-Null

# 3. 运行 Gradio 开发模式（自带监听器）
# 使用 gradio 命令行工具可以实现完美的监听和自动重启
.\venv\Scripts\python.exe -m gradio app.py
