# Instalación en Windows (HP Omen 16)

Guía paso a paso para dejar el proyecto funcionando en Windows 11 desde cero: búsqueda (embeddings en CPU), generación con Ollama en la GPU (RTX 3070 Ti, 8 GB) e interfaz web.

Todos los comandos se escriben en **PowerShell** (menú Inicio → escribe `PowerShell` → *Windows PowerShell*). No hace falta abrirlo como administrador.

## Por qué Windows nativo y no WSL

No hace falta WSL (Linux dentro de Windows):

- Todas las dependencias de Python (`torch`, `sentence-transformers`, `faiss-cpu`, `pandas`, `pyarrow`) tienen versión oficial para Windows y `uv` las instala sin compilar nada.
- Los embeddings se calculan en la **CPU**, así que Python no necesita CUDA.
- La **GPU solo la usa Ollama**, que tiene instalador oficial para Windows y trae su propio soporte para tarjetas NVIDIA.

Con WSL habría que instalar y mantener otro sistema y pasar la GPU a Linux, sin ganar nada.

## 1. Copiar el proyecto

1. Descomprime `stripe-rag-WINDOWS.zip` (clic derecho → *Extraer todo*) y deja la carpeta `stripe-rag` en una ruta **corta y sin tildes ni espacios**, por ejemplo `C:\TFG\stripe-rag`.
   Evita el Escritorio, *Documentos* sincronizados con OneDrive y rutas con caracteres como `ñ` o `á` (algunas librerías, como `faiss`, fallan con ellas).
2. El zip no trae `.venv` (es del Mac y no sirve en Windows) ni el índice: los crea `setup.ps1`. Sí trae en `out\` los **resultados ya medidos en el Mac**, que no se pueden regenerar sin repetir horas de evaluación: respuestas de Gemma y Qwen (`out\results\`), veredictos del juez (`out\judge\`), taxonomía de errores (`out\analysis\`) y resultados de cada experimento (`out\experiments\`). **No borres `out\`.**
3. Abre PowerShell y entra en la carpeta:

   ```powershell
   cd C:\TFG\stripe-rag
   ```

## 2. Instalar uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Cierra PowerShell y ábrelo otra vez** (para que reconozca el comando) y comprueba:

```powershell
uv --version
```

No hace falta instalar Python: `uv` descarga solo la versión correcta (3.13, la de `.python-version`).

## 3. Instalar Ollama

1. Descarga el instalador de <https://ollama.com/download/windows> y ejecútalo.
2. Al terminar, Ollama queda en marcha en segundo plano: verás el icono de la llama junto al reloj.
3. Cierra y abre PowerShell y comprueba:

   ```powershell
   ollama --version
   ```

No descargues el modelo a mano: lo hace `setup.ps1` en el paso 5. Para saber si usa la GPU, mira el paso 5.

## 4. Permitir ejecutar los scripts

Windows no deja ejecutar scripts `.ps1` por defecto. Una sola vez:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Responde `S` (o `Y`) si pregunta. Solo afecta a tu usuario.

Si has copiado el proyecto desde un `.zip` descargado, un correo o OneDrive, Windows marca los ficheros como "descargados de Internet" y los seguirá bloqueando. Desbloquéalos desde la carpeta del proyecto:

```powershell
Get-ChildItem -Recurse -Filter *.ps1 | Unblock-File
```

## 5. Preparar todo con `setup.ps1`

```powershell
.\scripts\windows\setup.ps1
```

Hace, en orden, y se para con un mensaje en rojo si algo falla:

1. Comprueba `uv`.
2. `uv sync`: crea `.venv` e instala las dependencias (la primera vez, varios minutos).
3. Comprueba que Ollama está en marcha (si no, intenta arrancarlo), descarga el modelo de `configs/ollama.yaml` (`qwen2.5-coder:7b`, unos 4.7 GB), le hace una pregunta corta y muestra `ollama ps`.
4. Construye el índice: `rag.extract`, `rag.chunk` y `rag.index` (la primera vez descarga de Hugging Face el modelo de embeddings y el tokenizador, unos 150 MB).
5. Pasa los tests (`uv run pytest`).

**Comprobar la GPU.** Al final del paso 3 sale la tabla de `ollama ps`. Fíjate en la columna `PROCESSOR`:

```text
NAME                ID              SIZE      PROCESSOR    UNTIL
qwen2.5-coder:7b    ...             6.0 GB    100% GPU     4 minutes from now
```

- `100% GPU`: perfecto.
- `xx%/yy% CPU/GPU`: el modelo no cabe entero en los 8 GB y va más lento. Mira *Solución de problemas*.

Puedes repetirlo cuando quieras con una pregunta en otra ventana y `ollama ps` mientras responde.

Opciones: `-SkipOllama` (sin Ollama ni modelo) y `-SkipTests`.

## 6. Comprobar que todo funciona

Después de `setup.ps1`, ejecuta:

```powershell
$env:PYTHONUTF8 = "1"
uv run python -m rag.eval_retrieval --dataset data/evals/dev_charge.jsonl --object-filter
uv run python -m rag.eval_retrieval --dataset data/evals/dev_objects.jsonl --object-filter
```

Checklist (deben salir **los mismos números** que en el Mac):

- [ ] `rag.extract`: `spec 2026-09-30.endive -> 3535 filas`
- [ ] `rag.chunk`: `hierarchical: 744 chunks` y `pasan del limite: 0`
- [ ] `rag.index`: `744 chunks indexados ... (384 dimensiones)`
- [ ] Tests: todos en verde (`passed`, ningún `failed`)
- [ ] `dev_charge`: `Preguntas evaluadas: 26`, `recall_at_5: 0.885`, `mrr: 0.780`
- [ ] `dev_objects`: `Preguntas evaluadas: 30`, `recall_at_5: 0.933`, `mrr: 0.774`
- [ ] `ollama ps` muestra `100% GPU` (o casi)
- [ ] La interfaz abre en <http://127.0.0.1:7860> y responde una pregunta

Si la búsqueda da números algo distintos en el tercer decimal, puede ser una diferencia numérica de la CPU; si la diferencia es grande, revisa que el índice se ha reconstruido con `setup.ps1`.

## 7. Usar la interfaz

```powershell
.\scripts\windows\run_app.ps1
```

Cuando salga `Running on local URL`, abre <http://127.0.0.1:7860> en el navegador. Para pararla, `Ctrl+C` en la ventana de PowerShell. Si el puerto está ocupado: `.\scripts\windows\run_app.ps1 -Port 7861`.

También puedes hacer una pregunta suelta desde la terminal:

```powershell
uv run python -m rag.generate "What currency does a charge use?"
```

## 8. Evaluar la generación

```powershell
.\scripts\windows\eval_generation.ps1
```

- Por defecto usa los conjuntos de desarrollo (`dev_charge` y `dev_objects`).
- Los resultados se guardan en `out\results\gen_*.jsonl` pregunta a pregunta.
- **Se puede interrumpir** con `Ctrl+C` y **continuar** después ejecutando el mismo comando: se salta las preguntas que ya tienen respuesta.
- Prueba rápida con pocas preguntas: `.\scripts\windows\eval_generation.ps1 -Limit 3`.
- Otro conjunto: `.\scripts\windows\eval_generation.ps1 -Dataset data\evals\control.jsonl`.

Todavía **no se sabe cuánto tarda cada pregunta** en el Omen: hay que medirlo. El script dice al final cuántos minutos ha tardado; apúntalo junto al número de preguntas para la memoria. Haz primero la prueba con `-Limit 3` para tener una idea.

Mientras evalúa, conviene tener el portátil enchufado y en modo de rendimiento (OMEN Gaming Hub), porque con batería la GPU va mucho más lenta.

## 9. Repetir los experimentos de búsqueda

No usan Ollama.

```powershell
.\scripts\windows\run_experiments.ps1                  # todos (tarda)
.\scripts\windows\run_experiments.ps1 E1_sin_copias    # solo uno
uv run python scripts/failures.py E1_sin_copias semantic+objfilter   # preguntas falladas
```

El resumen queda en `out\experiments\summary.md`. Las variantes E4 a E8 descargan modelos extra (unos 2 GB). El zip no trae los índices de cada variante, así que la primera vez se construyen (unos minutos por variante); después se reutilizan.

Las tablas de control, test y test 2 de la memoria (capítulo 5) se reproducen con:

```powershell
$env:PYTHONUTF8 = "1"
uv run python scripts/heldout.py        # escribe out\experiments\heldout_<conjunto>.md
```

## 10. Otras herramientas

Leen resultados ya guardados en `out\`, así que funcionan aunque no repitas nada:

| Para | Comando |
|---|---|
| Ver las preguntas falladas de una variante | `uv run python scripts/failures.py E1_sin_copias semantic+objfilter` |
| Taxonomía de errores de un modelo | `uv run python scripts/error_taxonomy.py --model google/gemma-4-e4b` |
| Resumen del juez | abrir `out\judge\summary.md` |

`scripts/judge.py prepare` y `score` solo hacen falta si se vuelven a generar respuestas: `prepare` prepara las respuestas anonimizadas, otro modelo las juzga y `score` calcula el resumen.

**Usar Gemma en Windows (opcional).** Los números de la memoria son de Gemma 4 E4B con LM Studio. Si quieres usarla también en el Omen, instala LM Studio (<https://lmstudio.ai>), descarga `google/gemma-4-e4b`, arranca su servidor local (pestaña *Developer* → *Start server*, puerto 1234) y usa la configuración por defecto en lugar de la de Ollama:

```powershell
$env:RAG_CONFIG = "configs/default.yaml"
.\scripts\windows\run_app.ps1
```


## Solución de problemas

**Ollama no está en marcha** (`Ollama no responde en http://127.0.0.1:11434`). Abre *Ollama* desde el menú Inicio y espera a que salga el icono de la llama. Comprueba con `ollama list`. Si `ollama` no se reconoce como comando, cierra y abre PowerShell (o reinstala Ollama).

**El modelo no está descargado** (`model "qwen2.5-coder:7b" not found`). Ejecuta `ollama pull qwen2.5-coder:7b` o vuelve a lanzar `setup.ps1`. Si cambias `llm.model` en `configs/ollama.yaml`, descarga el modelo nuevo igual.

**El modelo no cabe en los 8 GB de la GPU** (`ollama ps` muestra parte en CPU, o va muy lento). Prueba, en este orden:

1. Cierra juegos, el navegador con muchas pestañas u otros programas que usen la GPU.
2. En `configs/ollama.yaml`, baja `llm.num_ctx` de `8192` a `4096`.
3. Usa una cuantización más pequeña, por ejemplo `ollama pull qwen2.5-coder:7b-instruct-q3_K_M`, y pon ese nombre en `llm.model`. Anota en la memoria qué modelo has usado.

**La primera vez va muy lento.** Es normal: `uv sync` descarga las librerías (torch pesa bastante) y la primera búsqueda descarga los modelos de embeddings de Hugging Face (quedan en `C:\Users\<tu usuario>\.cache\huggingface`). Las siguientes veces es mucho más rápido. El aviso de Hugging Face sobre `symlinks` o sobre `HF_TOKEN` no es un error.

**Errores de codificación** (`UnicodeEncodeError`, `UnicodeDecodeError`, `charmap codec`, o caracteres raros como `â†’`). Windows usa `cp1252` por defecto y los chunks tienen caracteres como `→`. Los scripts `.ps1` ya ponen `PYTHONUTF8=1`. Si ejecutas `uv run ...` a mano, antes pon en esa ventana:

```powershell
$env:PYTHONUTF8 = "1"
```

Para que valga siempre: *Configuración → Sistema → Información → Configuración avanzada del sistema → Variables de entorno* → nueva variable de usuario `PYTHONUTF8` con valor `1`.

**El puerto 7860 está ocupado** (`El puerto 7860 ya esta en uso` o `Cannot find empty port`). Seguramente la interfaz ya está abierta en otra ventana: ciérrala con `Ctrl+C`. O usa otro puerto: `.\scripts\windows\run_app.ps1 -Port 7861`.

**Fallos al importar `torch` o `faiss`** (`DLL load failed`, `WinError 126`, `fbgemm.dll`). Instala el *Microsoft Visual C++ Redistributable* (x64) desde <https://aka.ms/vs/17/release/vc_redist.x64.exe>, reinicia y vuelve a lanzar `setup.ps1`. Si aparece `OMP: Error #15 ... libiomp5md.dll already initialized`, pon `$env:KMP_DUPLICATE_LIB_OK = "TRUE"` en esa ventana y repite el comando. Si `faiss` no puede abrir el índice, comprueba que la ruta del proyecto no tiene tildes ni `ñ`.

**Rutas demasiado largas** (`Filename too long`, `No such file or directory` dentro de `.venv` durante `uv sync`). Pon el proyecto en una ruta corta como `C:\TFG\stripe-rag`. Si sigue pasando, activa las rutas largas de Windows (PowerShell **como administrador**):

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

y reinicia.

**`uv sync` tarda muchísimo o falla al copiar ficheros.** El antivirus (Windows Defender) revisa cada fichero que se instala. Espera a que termine; si se repite, puedes añadir la carpeta del proyecto y `C:\Users\<tu usuario>\AppData\Local\uv` como exclusiones en *Seguridad de Windows → Protección antivirus → Exclusiones* (decisión tuya; quítalas después si prefieres). Si sale `Failed to hardlink files`, ejecuta `$env:UV_LINK_MODE = "copy"` y repite `uv sync`.

**`no se puede cargar el archivo ... porque la ejecución de scripts está deshabilitada`.** Repite el paso 4. Como alternativa, sin cambiar nada: `powershell -ExecutionPolicy Bypass -File .\scripts\windows\setup.ps1`.

**Has movido el proyecto de carpeta y `uv` falla.** Borra la carpeta `.venv` y ejecuta otra vez `setup.ps1`.
