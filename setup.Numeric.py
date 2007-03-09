#!/usr/bin/env python
#
# To install PyNGL:
#
#     python setup.py install
#

import os,sys
from os.path import join
from distutils.core import setup

#
# Get the root of where PyNGL and the extra PyNGL
# data files (fontcaps, graphcaps, map databases, example
# scripts, etc.) will live.
#
pyngl_pkg_name  = "PyNGL"

pkgs_pth        = join(sys.prefix,'lib','python'+sys.version[:3],\
                       'site-packages')
pyngl_dir       = join(pkgs_pth,pyngl_pkg_name)
pyngl_ncarg_dir = join(pyngl_dir,'ncarg')
pyngl_ndata_dir = join(pyngl_dir,'ncarg','data')
python_bin_dir  = join(sys.prefix,'bin')

#
# List all the extra files that need to be installed with PyNGL.
# These files include example PyNGL scripts, data for the scripts,
# fonts, map databases, colormaps, and other databases.
#

cwd = os.getcwd()
os.chdir(pyngl_pkg_name)
pyngl_ncarg_files = []
for root, dirs, files in os.walk('ncarg'):
  for name in files:
    pyngl_ncarg_files.append((os.path.join(pyngl_dir,root), \
                             [os.path.join(pyngl_pkg_name,root,name)]))
os.chdir(cwd)

pynglex_files  = os.listdir(join(pyngl_pkg_name,'ncarg','pynglex'))
bin_files      = os.listdir('bin')

for i in xrange(len(pynglex_files)):
  pynglex_files[i] = join(pyngl_pkg_name,'ncarg','pynglex',pynglex_files[i])

for i in xrange(len(bin_files)):
  bin_files[i] = join('bin',bin_files[i])

py_files = ['Ngl.py','Nio.py','hlu.py','__init__.py','pyngl_version.py',\
            'pynio_version.py']
for i in xrange(len(py_files)):
  py_files[i] = join(pyngl_pkg_name,py_files[i])

if sys.platform == "cygwin":
  so_files = ['_hlu.dll','nio.dll']
else:
  so_files = ['_hlu.so','nio.so']

for i in xrange(len(so_files)):
  so_files[i] = join(pyngl_pkg_name,so_files[i])

DATA_FILES = pyngl_ncarg_files
DATA_FILES.append(('bin',                   bin_files))
DATA_FILES.append((pkgs_pth,                ['PyNGL.pth']))
DATA_FILES.append((pyngl_dir,               so_files))
DATA_FILES.append((pyngl_dir,               py_files))
DATA_FILES.append((join(pyngl_ncarg_dir,'pynglex'),  pynglex_files))

setup (name = pyngl_pkg_name,
       version=XXXX,
       author='Dave Brown, Fred Clare, Mary Haley',
       author_email='dbrown@ucar.edu,fred@ucar.edu,haley@ucar.edu',
       maintainer = 'Mary Haley',
       maintainer_email = 'haley@ucar.edu',
       description = '2D visualization library',
       long_description = "PyNGL is a Python language module designed for publication-quality visualization of data. PyNGL stands for 'Python Interface to the NCL Graphics Libraries,' and it is pronounced 'pingle.' It now contains the 'Nio' module, which enables NetCDF-like access for NetCDF (rw), HDF (rw), GRIB (r), and CCM (r) data files",
       url = 'http://www.pyngl.ucar.edu/',
       data_files = DATA_FILES
      )
