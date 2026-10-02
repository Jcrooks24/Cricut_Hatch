# Creates double-clickable "Hatch" shortcuts (Desktop + Start Menu) that launch
# the app with pythonw.exe (no console window) using the repo's icon.
# Run:  powershell -ExecutionPolicy Bypass -File install_shortcut.ps1

$ErrorActionPreference = "Stop"
$repo   = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $repo "hatch_app.py"
$icon   = Join-Path $repo "assets\hatch.ico"

# Prefer a windowless pythonw.exe; fall back to python.exe.
$pyw = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\pythonw.exe"
if (-not (Test-Path $pyw)) {
    $cmd = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($cmd) { $pyw = $cmd.Source } else { $pyw = (Get-Command python.exe).Source }
}

$ws = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath('Desktop'),
                   [Environment]::GetFolderPath('Programs'))) {
    $lnk = Join-Path $dir "Hatch.lnk"
    $sc  = $ws.CreateShortcut($lnk)
    $sc.TargetPath       = $pyw
    $sc.Arguments        = "`"$script`""
    $sc.WorkingDirectory = $repo
    $sc.IconLocation     = "$icon,0"
    $sc.Description       = "Hatch - photo to pen-plotter SVG"
    $sc.WindowStyle       = 1
    $sc.Save()
    Write-Host "Created: $lnk"
}
Write-Host "Done. Double-click 'Hatch' on your Desktop, or find it in the Start menu."
