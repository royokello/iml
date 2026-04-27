from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent


setup(
    name="dequantize_from_affine_low_cuda",
    ext_modules=[
        CUDAExtension(
            name="dequantize_from_affine_low_cuda",
            sources=[
                str(ROOT / "dequantize_from_affine_low_extension.cpp"),
                str(ROOT / "dequantize_from_affine_low.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_61,code=sm_61",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
