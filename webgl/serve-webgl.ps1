# Minimal static file server (no node/python needed) for testing the UnitEye web pipeline.
$port = 8123
$root = "C:\Users\Mark\Desktop\uniteye\webgl"
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
        $file = Join-Path $root $path
        if (Test-Path $file -PathType Leaf) {
            $bytes = [System.IO.File]::ReadAllBytes($file)
            $ext = [System.IO.Path]::GetExtension($file).ToLower()
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
