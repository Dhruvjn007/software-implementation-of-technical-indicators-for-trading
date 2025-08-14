from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Compiler optimization flags
extra_compile_args = [
    "-O3",           # Maximum optimization
    "-ffast-math",   # Fast floating point math
    "-march=native", # Optimize for current CPU
    "-funroll-loops", # Loop unrolling
    "-ftree-vectorize" # Auto-vectorization
]

extensions = [
    Extension(
        "macd_fast",
        ["macd_fast.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        language="c"
    )
]

setup(
    name="macd_fast",
    ext_modules=cythonize(extensions, 
                         compiler_directives={
                             'language_level': 3,
                             'boundscheck': False,
                             'wraparound': False,
                             'cdivision': True,
                             'embedsignature': True
                         }),
    zip_safe=False,
)