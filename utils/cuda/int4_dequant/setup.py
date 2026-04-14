from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent


setup(
    name="int4_dequant_cuda",
    ext_modules=[
        CUDAExtension(
            name="int4_dequant_cuda",
            sources=[
                str(ROOT / "int4_dequant_extension.cpp"),
                str(ROOT / "int4_dequant_lut.cu"),
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
