# Arranca la interfaz web (Gradio).
#
# Uso (desde cualquier carpeta):
#   .\scripts\windows\run_app.ps1
#   .\scripts\windows\run_app.ps1 -Port 7861    # si el puerto 7860 esta ocupado
#
# Para pararla: Ctrl+C en esta ventana.
param(
    [int]$Port = 7860,
    [string]$ListenHost = "127.0.0.1"
)

. (Join-Path $PSScriptRoot "_common.ps1")

Write-Step "Comprobaciones"
Assert-Uv
Assert-Index
$backend = Get-ConfigValue "llm" "backend"
if ($backend -eq "ollama") {
    Assert-Ollama
    $model = Get-ConfigValue "llm" "model"
    Assert-OllamaModel $model
} else {
    $model = Get-ConfigValue "llm" "model"
    Write-Ok "Backend $backend ($model): asegurate de que el servidor esta en marcha (LM Studio: puerto 1234)"
}

$busy = $null
if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
    $busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
}
if ($busy) {
    Stop-WithError ("El puerto $Port ya esta en uso (quiza la interfaz ya esta abierta en otra ventana). " +
        "Cierra esa ventana o usa otro puerto: .\scripts\windows\run_app.ps1 -Port 7861")
}

Write-Step "Arrancando la interfaz"
Write-Host "    Cuando aparezca 'Running on local URL', abre en el navegador: http://${ListenHost}:$Port"
Write-Host "    Para pararla pulsa Ctrl+C en esta ventana."
Invoke-Native -Exe "uv" -Arguments @("run", "python", "-m", "rag.app", "--host", $ListenHost, "--port", "$Port") `
    -Description "rag.app"
