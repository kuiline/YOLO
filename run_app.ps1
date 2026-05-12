# 植物叶片病害检测系统 - 启动脚本 (支持热更新与自动打开浏览器)
$env:NO_PROXY = "127.0.0.1,localhost,*"
$env:no_proxy = "127.0.0.1,localhost,*"

$Root = $PSScriptRoot
Set-Location $Root

# 检查环境
if (-not (Test-Path "$Root\venv\Scripts\python.exe")) {
    Write-Host "❌ 错误: 未找到虚拟环境 venv" -ForegroundColor Red
    Read-Host "请先运行安装脚本或配置环境。按回车退出..."
    exit 1
}

$Url = "http://127.0.0.1:7860"

# 使用后台作业在 3 秒后打开浏览器（给后端启动留出时间）
Start-Job -ScriptBlock {
    param($u)
    Start-Sleep -Seconds 3
    Start-Process $u
} -ArgumentList $Url | Out-Null

Write-Host "----------------------------------------------------" -ForegroundColor Cyan
Write-Host "🚀 正在启动系统 [开发热更新模式]" -ForegroundColor Green
Write-Host "🌐 访问地址: $Url" -ForegroundColor Cyan
Write-Host "🔥 修改 app.py 或其他相关源码后，系统将自动重新加载。" -ForegroundColor Yellow
Write-Host "----------------------------------------------------" -ForegroundColor Cyan

# 使用 gradio CLI 启动以支持代码热更新 (Hot Reload)
# 如果只需要运行而不监控代码变化，可以使用: & "$Root\venv\Scripts\python.exe" "$Root\app.py"
& "$Root\venv\Scripts\gradio.exe" "$Root\app.py"

Read-Host "程序已退出。按回车关闭窗口..."
