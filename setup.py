#!/usr/bin/env python
#
# To build and/or install PyNGL:
#
#   python setup.py install
#
import sys,os
import shutil
from distutils.core import setup, Extension

try:
  path = os.environ["USE_NUMPY"]
  HAS_NUM = 2
except:
  HAS_NUM = 1

use_cvs = os.environ.get('USE_CVS')

# Get version info.

execfile('pyngl_version.py')
pyngl_version = version

if HAS_NUM == 2:
  DMACROS =  [('NeedFuncProto', None),('USE_NUMPY',None)]
else:
  DMACROS =  [('NeedFuncProto',None)]

#
# Get the root of where PyNGL will live, and where the extra PyNGL
# data files (fontcaps, graphcaps, map databases, example
# scripts, etc) will be installled.
#
pkgs_pth       = os.path.join(sys.prefix, 'lib', 'python'+sys.version[:3],
                              'site-packages')
pyngl_dir      = os.path.join(pkgs_pth, os.path.join('PyNGL','ncarg'))
pyngl_data_dir = os.path.join(pyngl_dir, 'data')

#
#
# Get root and various other directories of installed files.
#
ncl_root      = os.getenv("NCARG_ROOT")
ncl_bin       = os.path.join(ncl_root,'bin')
ncl_lib       = os.path.join(ncl_root,'lib')
ncl_inc       = [os.path.join(ncl_root,'include')]
ncl_ncarg_dir = os.path.join(ncl_lib,'ncarg')
ncl_data_dir  = os.path.join(ncl_ncarg_dir,'data')

#
# Get the list of pynglex *.py and *.res files. You have a choice of
# checking out a new directory from CVS, or using the "examples" directory.
# If you use the "examples" directory, make sure it doesn't have 
# any extraneous files, like *.ps files.
#
if use_cvs:
  pynglex_dir = "Scripts"
  os.system("/bin/rm -rf " + pynglex_dir)
  os.system("cvs co pynglex")
  pynglex_files = os.listdir(pynglex_dir)
else:
  pynglex_dir = "../examples"
  pynglex_files = os.listdir(pynglex_dir)
  if os.path.exists(os.path.join(pynglex_dir,'makefile')):
    pynglex_files.remove("makefile")

#
# Remove everything but *.py and *.res files from the list of files.
#
pynglex_files.remove("yMakefile")
pynglex_files.remove("CVS")
pynglex_files.remove("pynglex")

#
# Prepend the full directory path leading to files.
#
for i in xrange(len(pynglex_files)):
  pynglex_files[i] = os.path.join(pynglex_dir,pynglex_files[i])

#
# Gather up the executables we want to install as part of PyNGL.
# Get the NCAR Graphics executables from the installed location
# ($NCARG_ROOT).
#
bin_files  = ["ctrans","med","psplit"]
for i in xrange(len(bin_files)):
  bin_files[i] = os.path.join(ncl_bin,bin_files[i])
bin_files.append(os.path.join(pynglex_dir,'pynglex'))

#
# Location of system and NCARG include files and libraries.
#
# To include additional libraries, you can add them here, or, on
# the UNIX command line, you can type something like:
#
#  python setup.py build_ext -L/sw/lib
#
# You will then have to type "python setup.py install" separately to
# install the package.
#
ncl_and_sys_libs = [ncl_lib, "/usr/X11R6/lib"]

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
asc_files      = os.listdir(os.path.join(ncl_data_dir,'asc'))
dbin_files     = os.listdir(os.path.join(ncl_data_dir,'bin'))
cdf_files      = os.listdir(os.path.join(ncl_data_dir,'cdf'))
grb_files      = os.listdir(os.path.join(ncl_data_dir,'grb'))
colormap_files = os.listdir(os.path.join(ncl_ncarg_dir,'colormaps'))
fontcap_files  = os.listdir(os.path.join(ncl_ncarg_dir,'fontcaps'))
graphcap_files = os.listdir(os.path.join(ncl_ncarg_dir,'graphcaps'))
database_files = os.listdir(os.path.join(ncl_ncarg_dir,'database'))
database_files.remove("rangs")

#
# os.listdir doesn't include the relative directory path, so add it
# back here.
#
for i in xrange(len(asc_files)):
  asc_files[i] = os.path.join(ncl_data_dir,'asc',asc_files[i])

for i in xrange(len(dbin_files)):
  dbin_files[i] = os.path.join(ncl_data_dir,'bin',dbin_files[i])

for i in xrange(len(cdf_files)):
  cdf_files[i] = os.path.join(ncl_data_dir,'cdf',cdf_files[i])

for i in xrange(len(grb_files)):
  grb_files[i] = os.path.join(ncl_data_dir,'grb',grb_files[i])

for i in xrange(len(colormap_files)):
  colormap_files[i] = os.path.join(ncl_ncarg_dir,'colormaps',colormap_files[i])

for i in xrange(len(database_files)):
  database_files[i] = os.path.join(ncl_ncarg_dir,'database',database_files[i])

for i in xrange(len(fontcap_files)):
  fontcap_files[i] = os.path.join(ncl_ncarg_dir,'fontcaps',fontcap_files[i])

for i in xrange(len(graphcap_files)):
  graphcap_files[i] = os.path.join(ncl_ncarg_dir,'graphcaps',graphcap_files[i])

res_file = 'sysresfile'

#
# Gather up the *.py module files.
#
py_files= ['Ngl.py','hlu.py','__init__.py','pyngl_version.py']

#
# Here's the setup function.
#
setup (name = "PyNGL",
       version = pyngl_version,
       author="Fred Clare and Mary Haley",
       description = "2D visualization library",
       long_description = "PyNGL is a Python language module designed for publication-quality visualization of data. PyNGL stands for 'Python Interface to the NCL Graphics Libraries,' and it is pronounced 'pingle.'",
       url = "http://www.pyngl.ucar.edu/",
       package_dir = { 'PyNGL' : ''},
       scripts = bin_files,
       data_files = [(os.path.join(pyngl_dir,'pynglex'),pynglex_files),
                     (pkgs_pth,                ["PyNGL.pth"]),
                     (os.path.join(pkgs_pth,'PyNGL'), py_files),
                     (os.path.join(pyngl_data_dir,'asc'), asc_files),
                     (os.path.join(pyngl_data_dir,'bin'), dbin_files),
                     (os.path.join(pyngl_data_dir,'cdf'), cdf_files),
                     (os.path.join(pyngl_data_dir,'grb'), grb_files),
                     (os.path.join(pyngl_dir,'colormaps'),colormap_files),
                     (os.path.join(pyngl_dir,'database'), database_files),
                     (os.path.join(pyngl_dir,'fontcaps'), fontcap_files),
                     (os.path.join(pyngl_dir,'graphcaps'),graphcap_files),
                     (pyngl_dir, [res_file])],
       ext_package = 'PyNGL',
       ext_modules = [Extension('_hlu', 
                           ['Helper.c','hlu_wrap.c','gsun.c'],
                            define_macros = DMACROS,
                            library_dirs = ncl_and_sys_libs,
                            include_dirs = ncl_inc,
                            libraries = ["nfpfort", "hlu", "ncarg", "ncarg_gks", "ncarg_c", "ngmath", "X11", "g2c"])]
      )
