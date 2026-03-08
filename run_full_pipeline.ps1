# SAWN Full Automated Pipeline
# Runs: MoViNet Training -> Evaluation -> Verify outputs
# Run from: D:\sawn_project
# Usage: .\venv\Scripts\Activate.ps1; .\run_full_pipeline.ps1

$ErrorActionPreference = "Continue"
$DataDir   = "D:\sawn_project\New_SawnDataset"
$ModelDir  = "models\movinet"
$OutputDir = "outputs\evaluation"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " SAWN Full Pipeline Started: $(Get-Date)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# ── Step 1: Training (already running in background, skip if best model exists) ──
if (Test-Path "$ModelDir\movinet_best.pt") {
    Write-Host "[SKIP] Training already done. Found: $ModelDir\movinet_best.pt" -ForegroundColor Green
} else {
    Write-Host "[TRAIN] Starting MoViNet GPU training..." -ForegroundColor Yellow
    python scripts\train_movinet.py --data_dir $DataDir --epochs 30 --lr 1e-4
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Training failed! Exiting." -ForegroundColor Red
        exit 1
    }
    Write-Host "[TRAIN] Done. $(Get-Date)" -ForegroundColor Green
}

# ── Step 2: Evaluation on test set ───────────────────────────────────────────
Write-Host "`n[EVAL] Running evaluation on test set..." -ForegroundColor Yellow
python scripts\evaluate.py `
    --model    $ModelDir\movinet_best.pt `
    --test_dir $DataDir

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Evaluation had errors, but continuing..." -ForegroundColor DarkYellow
} else {
    Write-Host "[EVAL] Done. $(Get-Date)" -ForegroundColor Green
}

# ── Step 3: Verify outputs ────────────────────────────────────────────────────
Write-Host "`n[CHECK] Output files:" -ForegroundColor Yellow
@(
    "$ModelDir\movinet_best.pt",
    "$ModelDir\movinet_final.pt",
    "$ModelDir\confusion_matrix.png",
    "$ModelDir\training_curves.png"
) | ForEach-Object {
    if (Test-Path $_) {
        Write-Host "  OK  $_" -ForegroundColor Green
    } else {
        Write-Host "  MISSING  $_" -ForegroundColor Red
    }
}

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host " PIPELINE COMPLETE: $(Get-Date)" -ForegroundColor Cyan
Write-Host " Model weights: $ModelDir\" -ForegroundColor Cyan
Write-Host " To launch dashboard: python web\app.py" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
