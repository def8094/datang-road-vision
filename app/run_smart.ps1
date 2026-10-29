param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExecutable = ""
)
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $PythonExecutable = $pythonCommand.Source
}

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Python executable not found: $PythonExecutable. Activate the target Conda environment first."
}

# Prevent a user-level Python installation from overriding the activated environment.
$env:PYTHONNOUSERSITE = "1"
& $PythonExecutable (Join-Path $ProjectRoot "app\\main.py")
