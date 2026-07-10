param(
    [Parameter(Mandatory = $true)][string]$BuildDir,   # e.g. C:\Users\Mark\Desktop\uniteye_host63\BuildWebGL
    [Parameter(Mandatory = $true)][string]$TargetName  # e.g. build63 -> served at /build63/
)
# Copies a Unity WebGL build into the served webgl/ folder and wires in the UnitEye browser pipeline:
# - copies the build output
# - copies uniteye-core.js / uniteye-cv.js / uniteye-webgl-boot.js + models/eyemu.onnx next to it
# - injects the script tags into the generated index.html (before </head>)
$webglRoot = Split-Path $PSCommandPath -Parent
$dst = Join-Path $webglRoot $TargetName

if (-not (Test-Path (Join-Path $BuildDir "index.html"))) { Write-Error "No index.html in $BuildDir"; exit 1 }

if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
Copy-Item $BuildDir $dst -Recurse

Copy-Item (Join-Path $webglRoot "uniteye-core.js") $dst -Force
Copy-Item (Join-Path $webglRoot "uniteye-cv.js") $dst -Force
Copy-Item (Join-Path $webglRoot "uniteye-webgl-boot.js") $dst -Force
New-Item -ItemType Directory -Force -Path (Join-Path $dst "models") | Out-Null
Copy-Item (Join-Path $webglRoot "models\eyemu.onnx") (Join-Path $dst "models") -Force

$index = Join-Path $dst "index.html"
$html = [System.IO.File]::ReadAllText($index)
$inject = '<script src="uniteye-core.js"></script><script src="uniteye-webgl-boot.js"></script>'
if ($html -notmatch 'uniteye-core\.js') {
    $html = $html -replace '</head>', "$inject`r`n</head>"
    [System.IO.File]::WriteAllText($index, $html)
    Write-Output "POSTPROCESS_OK injected scripts into $index"
} else {
    Write-Output "POSTPROCESS_OK (already injected)"
}