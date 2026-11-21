# Minimal static file server (no node/python needed) for testing the UnitEye web pipeline.
$port = 8123
$root = [System.IO.Path]::GetFullPath($(if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }))
$rootPrefix = $root.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://localhost:$port/")
$listener.Start()
Write-Host "Serving $root on http://localhost:$port/ (default -> test.html)"
$mime = @{ ".html" = "text/html"; ".js" = "text/javascript"; ".json" = "application/json"; ".onnx" = "application/octet-stream"; ".wasm" = "application/wasm"; ".css" = "text/css"; ".mjs" = "text/javascript" }
while ($listener.IsListening) {
    try {
        $ctx = $listener.GetContext()
        $path = [System.Uri]::UnescapeDataString($ctx.Request.Url.LocalPath).TrimStart('/')
        if ([string]::IsNullOrEmpty($path)) { $path = "test.html" }
        $fullFile = [System.IO.Path]::GetFullPath((Join-Path $root $path))
        if ($fullFile.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase) -and (Test-Path $fullFile -PathType Leaf)) {
            $bytes = [System.IO.File]::ReadAllBytes($fullFile)
            $ext = [System.IO.Path]::GetExtension($fullFile).ToLower()
            if ($mime.ContainsKey($ext)) { $ctx.Response.ContentType = $mime[$ext] }
            $ctx.Response.Headers.Add("Access-Control-Allow-Origin", "*")
            $ctx.Response.ContentLength64 = $bytes.Length
            $ctx.Response.OutputStream.Write($bytes, 0, $bytes.Length)
        }
        else { $ctx.Response.StatusCode = 404 }
        $ctx.Response.Close()
    }
    catch { }
}
