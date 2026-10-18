$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
$source = Join-Path $project 'dist\PaperTranslator'
if (-not (Test-Path -LiteralPath (Join-Path $source 'PaperTranslator.exe'))) { throw 'Build the EXE first.' }
$installRoot = Join-Path $env:LOCALAPPDATA 'PaperTranslator'
$previousPath = Join-Path $installRoot 'installation.json'
$previous = if (Test-Path -LiteralPath $previousPath) { Get-Content -LiteralPath $previousPath -Raw | ConvertFrom-Json } else { $null }
$version = (& (Join-Path $project '.venv\Scripts\python.exe') -c 'import papertranslator;print(papertranslator.__version__)').Trim()
$appPath = Join-Path $installRoot ('app-v' + $version)
if (Test-Path -LiteralPath $appPath) {
    $appPath = Join-Path $installRoot ('app-v' + $version + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
Copy-Item -LiteralPath $source -Destination $appPath -Recurse
$shell = New-Object -ComObject WScript.Shell
$shortcutPath = Join-Path ([Environment]::GetFolderPath('Desktop')) '논문 번역기 (개발판).lnk'
if (Test-Path -LiteralPath $shortcutPath) {
    $existingShortcut = $shell.CreateShortcut($shortcutPath)
    $ownedShortcut = $previous -and ($previous.shortcut -eq $shortcutPath) -and ($existingShortcut.TargetPath -eq (Join-Path $previous.app 'PaperTranslator.exe'))
    if (-not $ownedShortcut) {
        $shortcutPath = Join-Path ([Environment]::GetFolderPath('Desktop')) ('논문 번역기 (개발판 ' + (Get-Date -Format 'HHmmss') + ').lnk')
    }
}
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $appPath 'PaperTranslator.exe'
$shortcut.WorkingDirectory = $appPath
$shortcut.Description = 'Local English to Korean academic PDF translator - development preview'
$shortcut.Save()
@{app=$appPath; shortcut=$shortcutPath; installed_at=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $installRoot 'installation.json') -Encoding utf8
Write-Output $appPath
