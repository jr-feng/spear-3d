from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="panoptic_postprocess_cuda",
    ext_modules=[
        CUDAExtension(
            name="panoptic_postprocess_cuda",
            sources=[
                "panoptic_postprocess_bindings.cpp",
                "panoptic_postprocess.cu",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
            extra_link_args=[
                "-Wl,-rpath,/home/nvidia/ssd/archiconda3/envs/openmmlab/lib",
                "-Wl,-rpath,/home/nvidia/ssd/archiconda3/envs/openmmlab/lib/python3.8/site-packages/torch/lib",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
