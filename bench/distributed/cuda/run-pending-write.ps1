# Run in Windows PowerShell with the CUDA Toolkit, VS 2022 C++ tools and CMake 3.24+.
$ErrorActionPreference = "Stop"
$fixture = Join-Path $PSScriptRoot "../fixtures/iq3_s-t1-p0"
$build = Join-Path ([System.IO.Path]::GetTempPath()) "strata-distributed-sm89-vs2022"

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "CMake 3.24 or newer must be on PATH."
}
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "CUDA Toolkit nvcc must be on PATH. Check nvcc --version in this PowerShell window."
}
cmake -S $PSScriptRoot -B $build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
if ($LASTEXITCODE -ne 0) { throw "CUDA / Visual Studio configuration failed." }
cmake --build $build --config Release --target pending_write
if ($LASTEXITCODE -ne 0) { throw "CUDA build failed." }
& (Join-Path $build "Release/pending_write.exe") (Join-Path $fixture "boundary") --metal-reference (Join-Path $fixture "layer04-materialized-residual.f32")
if ($LASTEXITCODE -ne 0) { throw "The CUDA comparison failed." }
Write-Output "PASS: layer 4 residual update matched the CPU and Metal references."
