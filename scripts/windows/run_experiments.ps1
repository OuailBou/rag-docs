# Repite los experimentos de busqueda de scripts\experiments.yaml (no usa Ollama).
# Cada variante guarda sus resultados en out\experiments\<variante>\ y el resumen
# en out\experiments\summary.md. Si una variante ya esta construida con la misma
# configuracion, no se vuelve a indexar: solo se evalua.
#
# Uso (desde cualquier carpeta):
#   .\scripts\windows\run_experiments.ps1                         # todas las variantes (tarda)
#   .\scripts\windows\run_experiments.ps1 E1_sin_copias           # solo una
#   .\scripts\windows\run_experiments.ps1 E1_sin_copias E2_troceado_fijo
#
# Nota: E4 a E8 descargan modelos extra (bge-base y rerankers, en total unos 2 GB).
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Variants = @()
)

. (Join-Path $PSScriptRoot "_common.ps1")

Write-Step "Comprobando uv"
Assert-Uv

$pyArgs = @("run", "python", "scripts/experiment.py") + @($Variants | Where-Object { $_ })
if ($Variants.Count -gt 0) {
    Write-Step "Ejecutando los experimentos: $($Variants -join ', ')"
} else {
    Write-Step "Ejecutando todos los experimentos (puede tardar bastante)"
}
$started = Get-Date
Invoke-Native -Exe "uv" -Arguments $pyArgs -Description "scripts/experiment.py"
$minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
Write-Host ""
Write-Ok "Terminado en $minutes minutos. Resumen en out\experiments\summary.md"
