import os
import glob
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


_ext_src_root = os.path.abspath('lib/pointnet2')
_ext_sources = glob.glob('{}/src/*.cpp'.format(_ext_src_root)) \
             + glob.glob('{}/src/*.cu'.format(_ext_src_root))
_ext_headers = "{}/include".format(_ext_src_root)

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            name='lib.pointnet2.ops', 
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format(_ext_headers)],
                "nvcc": ["-O2", "-I{}".format(_ext_headers)],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
