from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ext',
    ext_modules=[
        CUDAExtension(
            'ext',
            ['src/DTC.cu'],
            extra_compile_args={'cxx':['-g'], 'nvcc':['-O2', '--extended-lambda']},
            include_dirs=['src/']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)