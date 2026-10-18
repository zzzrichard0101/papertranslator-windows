$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $taskBuildEnvPath = $env:PATH
    try {
        # Prevent unrelated apps/runtime bundles on PATH from supplying DLLs.
        $env:PATH = "$env:WINDIR\System32;$env:WINDIR"
        & .\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --windowed --name PaperTranslator --paths src --collect-submodules papertranslator --collect-data rapidocr_onnxruntime --exclude-module PySide6.QtPdf --exclude-module PySide6.QtQml --exclude-module PySide6.QtQuick scripts/launcher.py
        if ($LASTEXITCODE -ne 0) { throw 'EXE build failed' }
    } finally { $env:PATH = $taskBuildEnvPath }
} finally { Pop-Location }
