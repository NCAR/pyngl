#!/usr/bin/env python
#
# To install PyNGL:
#
#     python setup.py install
#

import os
from distutils.core import setup, Extension

pyngl_inc = [os.getenv("NCARG_ROOT") + "/include"]
pyngl_lib = [os.getenv("NCARG_ROOT") + "/lib", "/usr/X11R6/lib", "/sw/lib"]
setup (name = "PyNGL",
       version="0.1.0b1",
       author="Fred Clare and Mary Haley",
       description = "2D visualization library",
       long_description = "PyNGL is a Python language module designed for publication-quality visualization of data. PyNGL stands for 'Python Interface to the NCL Graphics Libraries,' and it is pronounced 'pingle.'",
       url = "http://www.pyngl.ucar.edu/",
       packages = ['PyNGL'],
       ext_modules = [Extension('PyNGL._hlu', 
                           [os.path.join('', 'Helper.c'),
                            os.path.join('', 'hlu_wrap.c'), 
                            os.path.join('', 'gsun.c')],
                            define_macros=[('NeedFuncProto', None)],
                            include_dirs = pyngl_inc,
                            library_dirs = pyngl_lib,
                            libraries = ["hlu", "ncarg", "ncarg_gks", "ncarg_c", "ngmath", "X11", "g2c"])]
      )
