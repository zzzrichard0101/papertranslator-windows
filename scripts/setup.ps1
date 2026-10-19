param([ValidateSet('Light','Quality')][string]$Profile = 'Light')
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    & uv sync --frozen --group dev
    if ($LASTEXITCODE -ne 0) { throw 'Dependency setup failed' }
    & uv run python scripts/fetch_ocr_models.py
    if ($LASTEXITCODE -ne 0) { throw 'English OCR setup failed' }
    & uv run python scripts/bootstrap.py
    if ($LASTEXITCODE -ne 0) { throw 'Runtime/model setup failed' }
    $modelKey = 'hy7b'
    if ($Profile -eq 'Quality') {
        & uv run python scripts/fetch_qwen38.py
        if ($LASTEXITCODE -ne 0) { throw 'Optional model setup failed' }
        $modelKey = 'qwen38'
    }
    & uv run python scripts/select_model.py $modelKey
    if ($LASTEXITCODE -ne 0) { throw 'Model selection failed' }
} finally { Pop-Location }
