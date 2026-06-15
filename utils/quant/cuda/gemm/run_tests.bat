@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0..\..\..\..;%PYTHONPATH%

echo ===================================================
echo  Fused GEMM — Build + Test + Benchmark
echo ===================================================
echo.

REM 1. Build kernel
echo [1/4] Building kernel...
python setup.py build_ext --inplace
if %ERRORLEVEL% neq 0 (
    echo BUILD FAILED — check CUDA/nvcc setup
    exit /b 1
)
echo.

REM 2. Correctness tests
echo [2/4] Running correctness tests...
python -m utils.quant.cuda.gemm.evaluator
echo.

REM 3. Autotune + benchmark
echo [3/4] Running benchmark...
python -m utils.quant.cuda.gemm.benchmark
echo.

echo Done.
