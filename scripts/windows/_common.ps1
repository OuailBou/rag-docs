# Funciones compartidas por los scripts de Windows. No se ejecuta solo:
# los demas scripts lo cargan con ". (Join-Path $PSScriptRoot '_common.ps1')".
# Este fichero debe quedarse en ASCII (sin tildes ni enies).

$ErrorActionPreference = "Stop"

# Python en UTF-8 aunque Windows use cp1252 por defecto (los chunks llevan
# caracteres como flechas o guiones largos).
$env:PYTHONUTF8 = "1"
# Sin buffer: los mensajes de Python salen en cuanto se escriben (si no, PowerShell
# los recibe por una tuberia y pueden tardar en aparecer).
$env:PYTHONUNBUFFERED = "1"
# Hugging Face avisa de que Windows no tiene enlaces simbolicos; no es un error.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
# Que la consola muestre bien la salida UTF-8 de Python.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

# Raiz del proyecto: dos carpetas por encima de scripts\windows.
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
Set-Location -LiteralPath $ProjectRoot

# Los scripts de Windows usan Ollama: configs/ollama.yaml (salvo que RAG_CONFIG diga otra cosa).
if (-not $env:RAG_CONFIG) { $env:RAG_CONFIG = "configs/ollama.yaml" }

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Write-Ok([string]$Text) {
    Write-Host "    OK: $Text" -ForegroundColor Green
}

function Write-Warn([string]$Text) {
    Write-Host "    AVISO: $Text" -ForegroundColor Yellow
}

function Stop-WithError([string]$Text) {
    Write-Host ""
    Write-Host "ERROR: $Text" -ForegroundColor Red
    exit 1
}

# Ejecuta un programa externo (uv, ollama...) y para si devuelve un codigo de error.
# En Windows PowerShell 5.1, "Stop" convierte en error cualquier linea que el
# programa escriba en stderr (uv escribe ahi su progreso), por eso se relaja aqui.
function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$Arguments = @(),
        [string]$Description = ""
    )
    if (-not $Description) { $Description = "$Exe $($Arguments -join ' ')" }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @Arguments
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        Stop-WithError "Ha fallado: $Description (codigo $code)"
    }
}

# Como Invoke-Native, pero devuelve la ultima linea que el programa escribe en stdout.
function Get-NativeOutput {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$Arguments = @(),
        [string]$Description = ""
    )
    if (-not $Description) { $Description = "$Exe $($Arguments -join ' ')" }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $lines = & $Exe @Arguments
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        Stop-WithError "Ha fallado: $Description (codigo $code)"
    }
    $last = @($lines | Where-Object { "$_".Trim() -ne "" }) | Select-Object -Last 1
    return "$last".Trim()
}

# Busca uv. El instalador oficial lo deja en %USERPROFILE%\.local\bin y anade esa
# carpeta al PATH, pero solo en las terminales que se abran despues.
function Assert-Uv {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        $candidates = @(
            (Join-Path $env:USERPROFILE ".local\bin"),
            (Join-Path $env:USERPROFILE ".cargo\bin")
        )
        foreach ($dir in $candidates) {
            if (Test-Path (Join-Path $dir "uv.exe")) {
                $env:Path = "$dir;$env:Path"
                break
            }
        }
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Stop-WithError ("No se encuentra 'uv'. Instalalo con:`n" +
            "  powershell -ExecutionPolicy ByPass -c ""irm https://astral.sh/uv/install.ps1 | iex""`n" +
            "y despues cierra y vuelve a abrir la terminal. Mas detalles en docs\windows.md.")
    }
}

# Direccion del servidor de Ollama (por defecto http://127.0.0.1:11434).
function Get-OllamaUrl {
    $value = $env:OLLAMA_HOST
    if (-not $value) { return "http://127.0.0.1:11434" }
    if ($value -notmatch "^https?://") { $value = "http://$value" }
    $value = $value -replace "0\.0\.0\.0", "127.0.0.1"
    if ($value -notmatch ":\d+/?$") { $value = $value.TrimEnd("/") + ":11434" }
    return $value.TrimEnd("/")
}

function Test-OllamaRunning {
    try {
        Invoke-RestMethod -Uri "$(Get-OllamaUrl)/api/version" -TimeoutSec 5 | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Busca el programa ollama. Tras instalarlo, el PATH nuevo solo vale en terminales nuevas.
function Find-Ollama {
    if (Get-Command ollama -ErrorAction SilentlyContinue) { return $true }
    $dir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
    if (Test-Path (Join-Path $dir "ollama.exe")) {
        $env:Path = "$dir;$env:Path"
        return $true
    }
    return $false
}

# Comprueba que Ollama esta instalado y en marcha. Si no esta en marcha, intenta arrancarlo.
function Assert-Ollama {
    if (-not (Find-Ollama)) {
        Stop-WithError ("No se encuentra Ollama. Descargalo de https://ollama.com/download/windows, " +
            "instalalo y vuelve a abrir la terminal.")
    }
    if (Test-OllamaRunning) {
        Write-Ok "Ollama esta en marcha en $(Get-OllamaUrl)"
        return
    }
    Write-Warn "Ollama no responde. Intentando arrancarlo..."
    $app = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
    if (Test-Path $app) {
        Start-Process -FilePath $app | Out-Null
    } else {
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden | Out-Null
    }
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if (Test-OllamaRunning) {
            Write-Ok "Ollama arrancado en $(Get-OllamaUrl)"
            return
        }
    }
    Stop-WithError ("Ollama no responde en $(Get-OllamaUrl). Abre la aplicacion Ollama desde el menu Inicio " +
        "(debe salir el icono de la llama junto al reloj) y vuelve a ejecutar este script.")
}

# Lee un valor de la configuracion activa (RAG_CONFIG) con Python.
function Get-ConfigValue([string]$Section, [string]$Key) {
    $code = "import os, yaml; print(yaml.safe_load(open(os.environ['RAG_CONFIG'], encoding='utf-8'))['$Section']['$Key'])"
    return Get-NativeOutput -Exe "uv" -Arguments @("run", "python", "-c", $code) -Description "leer $Section.$Key de $env:RAG_CONFIG"
}

function Test-OllamaModel([string]$Model) {
    try {
        $tags = Invoke-RestMethod -Uri "$(Get-OllamaUrl)/api/tags" -TimeoutSec 10
    } catch {
        return $false
    }
    $names = @($tags.models | ForEach-Object { $_.name })
    return ($names -contains $Model) -or ($names -contains "${Model}:latest")
}

function Assert-OllamaModel([string]$Model) {
    if (-not (Test-OllamaModel $Model)) {
        Stop-WithError "El modelo '$Model' no esta descargado. Ejecuta: ollama pull $Model  (o scripts\windows\setup.ps1)"
    }
    Write-Ok "Modelo '$Model' disponible"
}

function Assert-Index {
    $out = if ($env:RAG_OUT) { $env:RAG_OUT } else { Join-Path $ProjectRoot "out" }
    foreach ($name in @("rag.faiss", "rag.metadata.jsonl", "rag.sqlite3")) {
        if (-not (Test-Path (Join-Path $out $name))) {
            Stop-WithError "Falta el indice ($out\$name). Ejecuta primero: .\scripts\windows\setup.ps1"
        }
    }
    Write-Ok "Indice encontrado en $out"
}
