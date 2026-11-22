# One-time setup of the demo environment in .venv (run from the repository root in PowerShell).
$ErrorActionPreference = "Stop"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r demo\requirements.txt
& .\.venv\Scripts\python.exe demo\patch_timm.py
Write-Host "Done. Start the demo with:  .\.venv\Scripts\python.exe -m streamlit run demo\app.py"
