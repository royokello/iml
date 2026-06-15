from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent


setup(
    name="gemm_fused_cuda",
    ext_modules=[
        CUDAExtension(
            name="gemm_fused_cuda",
            sources=[
                str(ROOT / "gemm_fused_extension.cpp"),
                str(ROOT / "gemm_fused.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_75,code=compute_75",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
