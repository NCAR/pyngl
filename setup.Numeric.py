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
# List directories that we need to get supplemental files from.
#
main_dirs    = ["ncarg","bin"]
ncarg_dirs   = ["data","colormaps","database","fontcaps","graphcaps"]
pynglex_dirs = ["pynglex"] 
ncarg_files  = ["sysresfile"]
pynglex_scp  = ["pynglex"]

#
#
# List all the extra files that need to be installed with PyNGL.
# These files include example PyNGL scripts, data for the scripts,
# fonts, map databases, colormaps, and other databases.
#
# os.listdir doesn't include the relative directory path
#
# We need a way to recursively list all files in the "ncarg"
# directory, rather than having to list each directory 
# individually. I think "os.walk" might be something to look into
# here.
#

ncarg_data = join('ncarg','data')
asc_files      = os.listdir(join(pyngl_pkg_name,ncarg_data,'asc'))
dbin_files     = os.listdir(join(pyngl_pkg_name,ncarg_data,'bin'))
cdf_files      = os.listdir(join(pyngl_pkg_name,ncarg_data,'cdf'))
grb_files      = os.listdir(join(pyngl_pkg_name,ncarg_data,'grb'))
colormap_files = os.listdir(join(pyngl_pkg_name,'ncarg','colormaps'))
database_files = os.listdir(join(pyngl_pkg_name,'ncarg','database'))
fontcap_files  = os.listdir(join(pyngl_pkg_name,'ncarg','fontcaps'))
graphcap_files = os.listdir(join(pyngl_pkg_name,'ncarg','graphcaps'))
pynglex_files  = os.listdir(join(pyngl_pkg_name,'ncarg','pynglex'))
bin_files      = os.listdir('bin')

#
# os.listdir doesn't include the relative directory path...
#
for i in xrange(len(asc_files)):
  asc_files[i] = join(pyngl_pkg_name,ncarg_data,'asc',asc_files[i])

for i in xrange(len(dbin_files)):
  dbin_files[i] = join(pyngl_pkg_name,ncarg_data,'bin',dbin_files[i])

for i in xrange(len(cdf_files)):
  cdf_files[i] = join(pyngl_pkg_name,ncarg_data,'cdf',cdf_files[i])

for i in xrange(len(grb_files)):
  grb_files[i] = join(pyngl_pkg_name,ncarg_data,'grb',grb_files[i])

for i in xrange(len(colormap_files)):
  colormap_files[i] = join(pyngl_pkg_name,'ncarg','colormaps',colormap_files[i])

for i in xrange(len(database_files)):
  database_files[i] = join(pyngl_pkg_name,'ncarg','database',database_files[i])

for i in xrange(len(fontcap_files)):
  fontcap_files[i] = join(pyngl_pkg_name,'ncarg','fontcaps',fontcap_files[i])

for i in xrange(len(graphcap_files)):
  graphcap_files[i] = join(pyngl_pkg_name,'ncarg','graphcaps',graphcap_files[i])

for i in xrange(len(pynglex_files)):
  pynglex_files[i] = join(pyngl_pkg_name,'ncarg','pynglex',pynglex_files[i])

for i in xrange(len(bin_files)):
  bin_files[i] = join('bin',bin_files[i])

res_file = [join(pyngl_pkg_name,'ncarg','sysresfile')]

py_files = ['Ngl.py','Nio.py','hlu.py','__init__.py','pyngl_version.py',\
            'pynio_version.py']
for i in xrange(len(py_files)):
  py_files[i] = join(pyngl_pkg_name,py_files[i])

so_files = ['_hlu.so','nio.so']
for i in xrange(len(so_files)):
  so_files[i] = join(pyngl_pkg_name,so_files[i])

setup (name = pyngl_pkg_name,
       version='1.0.0',
       author='Dave Brown, Fred Clare, Mary Haley',
       author_email='dbrown@ucar.edu,fred@ucar.edu,haley@ucar.edu',
       maintainer = 'Mary Haley',
       maintainer_email = 'haley@ucar.edu',
       description = '2D visualization library',
       long_description = "PyNGL is a Python language module designed for publication-quality visualization of data. PyNGL stands for 'Python Interface to the NCL Graphics Libraries,' and it is pronounced 'pingle.' It now contains the 'Nio' module, which enables NetCDF-like access for NetCDF (rw), HDF (rw), GRIB (r), and CCM (r) data files",
       url = 'http://www.pyngl.ucar.edu/',
       data_files = [('bin',                   bin_files),
                     (pkgs_pth,                ['PyNGL.pth']),
                     (pyngl_dir,               so_files),
                     (pyngl_dir,               py_files),
                     (pyngl_ncarg_dir,         res_file),
                     (join(pyngl_ndata_dir,'asc'), asc_files),
                     (join(pyngl_ndata_dir,'bin'), dbin_files),
                     (join(pyngl_ndata_dir,'cdf'), cdf_files),
                     (join(pyngl_ndata_dir,'grb'), grb_files),
                     (join(pyngl_ncarg_dir,'colormaps'),colormap_files),
                     (join(pyngl_ncarg_dir,'database'), database_files),
                     (join(pyngl_ncarg_dir,'fontcaps'), fontcap_files),
                     (join(pyngl_ncarg_dir,'graphcaps'),graphcap_files),
                     (join(pyngl_ncarg_dir,'pynglex'),  pynglex_files)]
      )
