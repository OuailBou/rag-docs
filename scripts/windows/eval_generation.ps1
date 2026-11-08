# Evalua la generacion de respuestas con Ollama sobre los conjuntos de desarrollo.
# Se puede interrumpir con Ctrl+C y continuar despues: al volver a lanzarlo,
# se salta las preguntas que ya tienen respuesta en out\results\gen_*.jsonl.
#
# Uso (desde cualquier carpeta):
#   .\scripts\windows\eval_generation.ps1                       # dev_charge + dev_objects
#   .\scripts\windows\eval_generation.ps1 -Limit 3              # solo 3 preguntas por conjunto (prueba rapida)
#   .\scripts\windows\eval_generation.ps1 -Dataset data\evals\control.jsonl
#   .\scripts\windows\eval_generation.ps1 -Dataset data\evals\dev_charge.jsonl, data\evals\dev_objects.jsonl
param(
    [string[]]$Dataset = @("data/evals/dev_charge.jsonl", "data/evals/dev_objects.jsonl"),
    [int]$Limit = 0
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

$pyArgs = @("run", "python", "-m", "rag.eval_generation")
foreach ($item in $Dataset) {
    foreach ($part in ($item -split ",")) {
        $ds = $part.Trim()
        if (-not $ds) { continue }
        if (-not (Test-Path -LiteralPath $ds)) { Stop-WithError "No existe el conjunto de preguntas: $ds" }
        $pyArgs += @("--dataset", $ds)
    }
}
if ($Limit -gt 0) { $pyArgs += @("--limit", "$Limit") }

Write-Step "Evaluando la generacion con $model"
Write-Host "    Puede tardar bastante. Puedes pararlo con Ctrl+C y continuar despues"
Write-Host "    ejecutando de nuevo este mismo comando."
$started = Get-Date
Invoke-Native -Exe "uv" -Arguments $pyArgs -Description "rag.eval_generation"
$minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
Write-Host ""
Write-Ok "Terminado en $minutes minutos. Resultados en out\results\gen_*.jsonl"
