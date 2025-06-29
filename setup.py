from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension, setup
setup(
    name="int8_gemm",
    ext_modules=[
        CUDAExtension(
            name="int8_gemm",
            sources=["cuda/int8_gemm.cu"],
            extra_compile_args={"cxx": [], "nvcc": ["-O3", "-gencode=arch=compute_61,code=sm_61"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
