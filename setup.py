from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np
import os

Options.docstrings = True

sourcefiles = ["miololib.pyx"]

os.environ["CXX"] = "clang"

ext = Extension('miolo', sourcefiles, language="c++",
        libraries=[],
        include_dirs=[np.get_include()])

setup(ext_modules=cythonize(ext,compiler_directives={'boundscheck':False,
                                                     'embedsignature':True}))