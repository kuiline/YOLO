# YOLO 训练启动脚本
# 用法: 直接双击或在 PowerShell 中运行 .\run_train.ps1

$Root = $PSScriptRoot
Set-Location $Root

if (-not (Test-Path "$Root\venv\Scripts\python.exe")) {
    Write-Host "Error: venv not found. Please run setup first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "  YOLO 叶片病害检测 - 训练启动" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "模型   : yolov8x.pt" -ForegroundColor Yellow
Write-Host "数据集 : datasets/data.yaml" -ForegroundColor Yellow
Write-Host "Epochs : 100" -ForegroundColor Yellow
Write-Host "Batch  : 8" -ForegroundColor Yellow
Write-Host "输出   : runs/detect/train_v8x/" -ForegroundColor Yellow
Write-Host ""
Write-Host "开始训练，按 Ctrl+C 可中断..." -ForegroundColor Green
Write-Host ""

& "$Root\venv\Scripts\python.exe" "$Root\train.py"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "训练完成！模型保存在 runs/detect/runs/detect/train_v8x/weights/best.pt" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "训练异常退出，exit code: $LASTEXITCODE" -ForegroundColor Red
}

Read-Host "Press Enter to exit"
