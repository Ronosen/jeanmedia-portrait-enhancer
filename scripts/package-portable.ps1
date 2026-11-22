param([string]$PythonRoot="$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python")
$ErrorActionPreference='Stop'
$projectRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$releaseRoot=Join-Path $projectRoot 'release'
$destination=Join-Path $releaseRoot ('JeanMedia-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
if(Test-Path -LiteralPath $destination){throw 'Release folder already exists.'}
New-Item -ItemType Directory -Path $destination | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'src-tauri\target\debug\jeanmedia.exe') -Destination $destination
New-Item -ItemType Directory -Path (Join-Path $destination 'python') | Out-Null
Get-ChildItem -LiteralPath $PythonRoot -File | Where-Object { $_.Extension -in '.exe','.dll' -or $_.Name -eq 'LICENSE.txt' } | Copy-Item -Destination (Join-Path $destination 'python')
function Copy-Tree([string]$From,[string]$To,[string[]]$Exclude=@()) {
    $copyArgs=@($From,$To,'/E','/NFL','/NDL','/NJH','/NJS','/NP')
    if($Exclude.Count){$copyArgs+=@('/XD'); $copyArgs+=$Exclude}
    & robocopy @copyArgs | Out-Null
    if($LASTEXITCODE -gt 7){throw "Copy failed: $From"}
}
Copy-Tree (Join-Path $PythonRoot 'DLLs') (Join-Path $destination 'python\DLLs')
Copy-Tree (Join-Path $PythonRoot 'Lib') (Join-Path $destination 'python\Lib') @('site-packages','__pycache__','test','tests','turtledemo','idlelib')
Copy-Tree (Join-Path $projectRoot '.worker-deps') (Join-Path $destination '.worker-deps') @('__pycache__','tests','test')
Copy-Tree (Join-Path $projectRoot 'worker') (Join-Path $destination 'worker') @('__pycache__','.runtime-cache','.cache')
foreach($name in @('README.md','SEPARATED-WORKFLOW.md','SCENE-INTENT.md','EDGE-REFINEMENT.md','EDGE-DEPENDENCIES.json','CHECKLIST.md','MODEL-MANIFEST.json','MODEL-DECISIONS.md','LICENSE-INVENTORY.json','USER-GUIDE.md','API.md','ACCEPTANCE.md','QA-INPAINTING.json','QA-ALL-IMAGES.md','ADAPTIVE-CLEANUP.md','QA-ADAPTIVE.json','MASKS-AND-BACKDROP.md','QA-V3.json','VERIFICATION.json')) {
    $source=Join-Path $projectRoot $name
    if(Test-Path -LiteralPath $source){Copy-Item -LiteralPath $source -Destination $destination}
}
$env:PYTHONPATH=Join-Path $destination '.worker-deps'
$diagnostic='{}' | & (Join-Path $destination 'python\python.exe') (Join-Path $destination 'worker\service.py') diagnostics
if($LASTEXITCODE -ne 0){throw 'Portable runtime self-test failed.'}
$diagnostic | Set-Content -Encoding utf8 (Join-Path $destination 'SELF-TEST.json')
$report=$diagnostic | ConvertFrom-Json
if(@($report.selfTest | Where-Object {-not $_.ok}).Count){throw 'Portable model self-test failed.'}
Write-Output $destination
