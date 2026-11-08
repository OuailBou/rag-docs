# Prepara el proyecto en Windows de principio a fin:
#   1. comprueba uv e instala las dependencias (uv sync)
#   2. comprueba Ollama y descarga el modelo de configs/ollama.yaml (llm.model)
#   3. construye el indice (extract -> chunk -> index)
#   4. pasa los tests
#
# Uso (desde cualquier carpeta):
#   .\scripts\windows\setup.ps1
#   .\scripts\windows\setup.ps1 -SkipOllama    # sin comprobar Ollama ni descargar el modelo
#   .\scripts\windows\setup.ps1 -SkipTests     # sin pasar los tests
param(
    [switch]$SkipOllama,
    [switch]$SkipTests
)

. (Join-Path $PSScriptRoot "_common.ps1")

Write-Host "Proyecto: $ProjectRoot"

Write-Step "1/5 Comprobando uv"
Assert-Uv
Invoke-Native -Exe "uv" -Arguments @("--version")

Write-Step "2/5 Instalando dependencias (uv sync). La primera vez tarda varios minutos."
Invoke-Native -Exe "uv" -Arguments @("sync") -Description "uv sync"
Write-Ok "Dependencias instaladas en .venv"

if ($SkipOllama) {
    Write-Step "3/5 Ollama: omitido (-SkipOllama)"
} else {
    Write-Step "3/5 Comprobando Ollama y el modelo"
    Assert-Ollama
    $model = Get-ConfigValue "llm" "model"
    $numCtx = Get-ConfigValue "llm" "num_ctx"
    Write-Host "    Modelo de ${env:RAG_CONFIG}: $model (num_ctx = $numCtx)"
    if (Test-OllamaModel $model) {
        Write-Ok "El modelo ya estaba descargado"
    } else {
        Write-Host "    Descargando $model (varios GB, puede tardar)..."
        Invoke-Native -Exe "ollama" -Arguments @("pull", $model) -Description "ollama pull $model"
        Write-Ok "Modelo descargado"
    }

    Write-Host "    Cargando el modelo para comprobar la GPU (la primera vez tarda un poco)..."
    $body = @{
        model   = $model
        prompt  = "Di solo: hola"
        stream  = $false
        options = @{ num_predict = 8; num_ctx = [int]$numCtx }
    } | ConvertTo-Json -Depth 5
    try {
        $reply = Invoke-RestMethod -Uri "$(Get-OllamaUrl)/api/generate" -Method Post -Body $body `
            -ContentType "application/json" -TimeoutSec 600
        $answer = "$($reply.response)".Trim()
        Write-Ok "El modelo responde: $answer"
    } catch {
        Write-Warn "El modelo no ha respondido: $($_.Exception.Message)"
    }
    Write-Host "    Estado de Ollama (columna PROCESSOR: lo ideal es '100% GPU'):"
    Invoke-Native -Exe "ollama" -Arguments @("ps")
}

Write-Step "4/5 Construyendo el indice. La primera vez descarga los modelos de embeddings."
Invoke-Native -Exe "uv" -Arguments @("run", "python", "-m", "rag.extract") -Description "rag.extract"
Invoke-Native -Exe "uv" -Arguments @("run", "python", "-m", "rag.chunk") -Description "rag.chunk"
Invoke-Native -Exe "uv" -Arguments @("run", "python", "-m", "rag.index") -Description "rag.index"
Write-Ok "Indice creado en out\"

if ($SkipTests) {
    Write-Step "5/5 Tests: omitidos (-SkipTests)"
} else {
    Write-Step "5/5 Pasando los tests"
    Invoke-Native -Exe "uv" -Arguments @("run", "pytest", "-q") -Description "pytest"
    Write-Ok "Tests correctos"
}

Write-Host ""
Write-Host "Todo listo." -ForegroundColor Green
Write-Host "  Interfaz:               .\scripts\windows\run_app.ps1"
Write-Host "  Evaluar la generacion:  .\scripts\windows\eval_generation.ps1"
Write-Host "  Experimentos:           .\scripts\windows\run_experiments.ps1"
Write-Host "  Comprobar la busqueda:  uv run python -m rag.eval_retrieval --dataset data/evals/dev_charge.jsonl --object-filter"
