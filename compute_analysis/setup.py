from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sim_ext',
    ext_modules=[
        CUDAExtension(
            'sim_ext',
            ['src/spmm_compare.cu'],
            extra_compile_args={'cxx':['-g'], 
                                'nvcc':['-O2', '-arch=sm_86', '--extended-lambda']},
            include_dirs=['src/'],
            libraries=['cuda']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)