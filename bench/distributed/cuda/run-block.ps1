# Build separately from the installed engine. The first build fetches the pinned ggml sources.
[CmdletBinding(DefaultParameterSetName="Run")]
param(
    [Parameter(Mandatory, ParameterSetName="Build")][switch]$BuildOnly,
    [Parameter(Mandatory, ParameterSetName="Run")][string]$Pack,
    [Parameter(Mandatory, ParameterSetName="Run")][string]$Shard,
    [Parameter(Mandatory, ParameterSetName="Run")][string]$Out
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$scratch = Join-Path $env:LOCALAPPDATA "StrataDistributed"
$build = Join-Path $scratch "block-sm89-vs2022"
foreach ($command in @("cmake", "git", "nvcc")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Missing $command on PATH. Use CMake 3.24+, Git and the CUDA Toolkit with VS 2022 C++ integration."
    }
}
if (-not $BuildOnly) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "Python 3 and the py launcher are required." }
    if (-not (Test-Path -LiteralPath $Pack -PathType Container)) { throw "Pack directory not found: $Pack" }
    if (-not (Test-Path -LiteralPath $Shard -PathType Leaf)) { throw "First GGUF shard not found: $Shard" }
    if (Test-Path -LiteralPath $Out) { throw "Choose a new output directory: $Out" }
}

# Fetch just the commit named by Strata, avoiding a full clone of llama.cpp's history.
$pin = ((Get-Content -LiteralPath (Join-Path $repo "third_party/ggml/VERSION.txt") -TotalCount 1) -split " ")[0]
if ($pin -notmatch "^[0-9a-f]{40}$") { throw "Invalid pinned llama.cpp revision" }
$llama = Join-Path $scratch ("llama-" + $pin.Substring(0, 12))
if (-not (Test-Path -LiteralPath (Join-Path $llama ".git"))) {
    git init -q $llama
    if ($LASTEXITCODE -ne 0) { throw "Could not initialise the ggml source directory." }
}
$head = git -C $llama rev-parse --verify --quiet HEAD
if ($head -ne $pin) {
    git -C $llama fetch --depth 1 https://github.com/ggml-org/llama.cpp.git $pin
    if ($LASTEXITCODE -ne 0) { throw "Failed to fetch pinned llama.cpp revision $pin." }
    git -C $llama checkout --detach $pin
    if ($LASTEXITCODE -ne 0) { throw "Could not use pinned llama.cpp revision $pin." }
}
cmake -S $repo -B $build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89 `
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF `
    -DSTRATA_BUILD_CUDA_BLOCK_PROBE=ON "-DSTRATA_GGML_DIR=$llama"
if ($LASTEXITCODE -ne 0) { throw "Configuration failed. Check VS 2022 C++ tools and CUDA integration." }
cmake --build $build --config Release --target strata_block_probe --parallel 4
if ($LASTEXITCODE -ne 0) { throw "The CUDA block probe failed to build." }
$engine = Join-Path $build "Release/strata_block_probe.exe"
if (-not (Test-Path -LiteralPath $engine -PathType Leaf)) { throw "Missing probe executable: $engine" }
if ($BuildOnly) { Write-Output "Built $engine"; return }

py -3 (Join-Path $PSScriptRoot "../run_cuda_block.py") --engine $engine --pack $Pack --shard $Shard --out $Out
if ($LASTEXITCODE -ne 0) { throw "CUDA block test failed. See the diagnostic above." }
