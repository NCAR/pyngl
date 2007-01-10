#!/usr/bin/env python
#
# This script is for building and installing the PyNGL analysis functions.
# They will get installed in the PAF package directory (for Numeric) and
# PAF_numpy (for numpy). Right now, I don't think the numpy version works.
#
# Eventually, this stuff will probably be combined with PyNGL, and become
# part of the PyNGL build system.
#
# To build and install the functions:
#
#   python setup.py install
#
# To build but not install:
#
#   python setup.py build_ext
#
# To create a binary distribution:
#
#   python setup.py bdist_dumb --relative
#
# If no environment variables are set, this script will create
# a Numeric version of PAF.
#
# There are four environment variables that if set, will change
# the behavior of this script:
#
#    USE_NUMPY     - Create a numpy version of PAF only.
#
#    USE_NUMERPY   - Create a Numeric *and* numpy version of PAF. This
#                    will create two packages: PAF and PAF_numpy.
#
import sys,os,re
import fileinput
import shutil
from distutils.core import setup, Extension

#
# Determine whether we want to build a Numeric and/or Numpy version
# of PAF.  If the environment variable USE_NUMPY is set, it will
# try to build a numpy version. USE_NUMPY doesn't need to be set to
# any value; it just has to be set.  If USE_NUMERPY is set, then
# both versions of PAF will be built, and the Numeric version will
# be put in package PAF, and the numpy version in package PAF_numpy.
#
# HAS_NUM will be set by this script depending on USE_NUMPY and USE_NUMERPY.
#
# HAS_NUM = 3 --> install both numpy and Numeric versions of PAF
# HAS_NUM = 2 --> install numpy version of PAF
# HAS_NUM = 1 --> install Numeric version of PAF
# HAS_NUM = 0 --> You're hosed as numpy or Numeric don't exist.
#
try:
  path = os.environ["USE_NUMERPY"]
  HAS_NUM = 3
except:
  try:
    path = os.environ["USE_NUMPY"]
    HAS_NUM = 2
  except:
    HAS_NUM = 1

#
# Test to make sure we actually the Numeric and/or numpy modules
# that we have requested.
#
if HAS_NUM > 1:
  try:
    import numpy
  except ImportError:
    print "Cannot find numpy; we'll try Numeric."
    HAS_NUM = 1

if HAS_NUM == 1 or HAS_NUM == 3:
  try:
    import Numeric
  except ImportError:
    print "Cannot find Numeric."
    HAS_NUM = HAS_NUM-1

if HAS_NUM == 3:
  array_modules = ['Numeric','numpy']
elif HAS_NUM == 2:
  array_modules = ['numpy']
elif HAS_NUM == 1:
  array_modules = ['Numeric']
else:
  print "Cannot find Numeric or numpy; good-bye!"
  exit

#
# Initialize some variables.
#
paf_vfile = "paf_version.py"            # PAF version file.
pkgs_pth  = os.path.join(sys.prefix, 'lib', 'python'+sys.version[:3],
                         'site-packages')

#
# Get root and various other directories of installed files.
#
ncl_root      = os.getenv("NCARG_ROOT")
ncl_bin       = os.path.join(ncl_root,'bin')
ncl_lib       = os.path.join(ncl_root,'lib')
ncl_inc       = os.path.join(ncl_root,'include')

ncl_and_sys_lib_paths = [ncl_lib, "/usr/X11R6/lib"]

if sys.platform == "darwin":
    dirs = ['/sw/lib','/Users/haley/lib/gcc-lib/i386-apple-darwin8.6.1/4.0.3']
    for dir in dirs:
      if(os.path.exists(dir)):
        ncl_and_sys_lib_paths.append(dir)

if sys.platform == "linux2" and os.uname()[-1] == "x86_64":
    ncl_and_sys_lib_paths.append('/usr/X11R6/lib64')

#
# Gather up the *.py module files.
#
py_files= ['NglA.py','__init__.py','paf_version.py']

EXTRA_LINK_ARGS = ""
LIBRARIES = ["nfpfort", "hlu", "ncarg", "ncarg_gks", "ncarg_c", "ngmath", "X11", "g2c"]

if sys.platform == "aix5":
    LIBRARIES.remove('g2c')
    LIBRARIES.append('xlf90')
#
# Special test for Intel Mac platform, which is using the g95 compiler
# and needs f95 loaded.
#
if sys.platform == "darwin":
    dir = '/Users/haley/lib/gcc-lib/i386-apple-darwin8.6.1/4.0.3'
    if dir in ncl_and_sys_lib_paths:
      LIBRARIES.remove('g2c')
      LIBRARIES.append('f95')

if sys.platform == "sunos5":
    os.environ["CC"]="/opt/SUNWspro/bin/cc"
    LIBRARIES.remove('g2c')
    LIBRARIES.append('f77compat')
    LIBRARIES.append('fsu')
    LIBRARIES.append('sunmath')

for array_module in array_modules:
  INCLUDE_PATHS = [ncl_inc]

#----------------------------------------------------------------------
#
# Create version file that contains version and array module info.
#
#----------------------------------------------------------------------
  if array_module == 'Numeric':
    from Numeric import  __version__ as array_module_version

    paf_pkg_name = 'PAF'
    paf_pth_file = [paf_pkg_name + '.pth']

    print '====> Installing the numeric version of PAF to the',paf_pkg_name,'package directory.'

    DMACROS       =  [('NeedFuncProto',None)]
  else:
    from numpy import __version__ as array_module_version

    paf_pkg_name = 'PAF_numpy'
    paf_pth_file = []              # No *.pth file for numpy package!

    print '====> Installing the numpy version of PAF to the',paf_pkg_name,'package directory.'

#
# For a numpy build, we need to point to the correct array "arrayobject.h"
# and set the USE_NUMPY macro for compiling the *.c files.
#
    INCLUDE_PATHS.insert(0,os.path.join(pkgs_pth,"numpy/core/include"))
    DMACROS =  [('NeedFuncProto', None),('USE_NUMPY',None)]

#----------------------------------------------------------------------
#
# Here are the instructions for compiling the "fplib.so" file.
#
#----------------------------------------------------------------------
  EXT_MODULES = [Extension('fplib', 
                 ['fplibmodule.c'],
                  define_macros   = DMACROS,
                  include_dirs    = INCLUDE_PATHS,
                  library_dirs    = ncl_and_sys_lib_paths,
                  libraries       = LIBRARIES)]
  paf_dir       = os.path.join(pkgs_pth, paf_pkg_name)

#
# This seems kludgy to me, but I need to make sure that if both 
# numpy and Numeric versions of PAF are being built, we clean
# the *.o files beforehand. This is because "setup" puts the *.o files
# in the same location (build/temp.xxxx/.) every time, regardless of which
# package we're building. Maybe there's a way to tell setup to put the
# *.o files in a different directory, but I haven't found it yet.
#
  if len(array_modules) > 1:
    print "====> Removing build's *.o and *.so files..."
    os.system("find build -name '*.o' -exec /bin/rm {} \;")
    os.system("find build -name '*.so' -exec /bin/rm {} \;")

#
# Get version number.
#
  if os.path.exists(paf_vfile):
    os.system("/bin/rm -rf " + paf_vfile)

  paf_version = open('version','r').readlines()[0].strip('\n')
  vfile = open(paf_vfile,'w')
  vfile.write("version = '%s'\n" % paf_version)
  vfile.write("array_module = '%s'\n" % array_module)
#
# The Ngl.py and Nio.py files use HAS_NUM to tell whether to use
# Numeric or numpy specific operations.
#
  if array_module == 'Numeric':
    vfile.write("HAS_NUM = 1\n")
  else:
    vfile.write("HAS_NUM = 2\n")

  vfile.write("array_module_version = '%s'\n" % array_module_version)
  vfile.close()

  DATA_FILES = [(pkgs_pth, paf_pth_file),(paf_dir,py_files)]

#
# Here's the setup function that will build and install everything.
#
  setup (name = paf_pkg_name,
         version = paf_version,
         author = "Fred Clare and Mary Haley",
         maintainer = "Mary Haley",
         maintainer_email = "haley@ucar.edu",
         description = "PyNGL Analysis Functions",
         long_description = "PyNGL is a Python language module designed for publication-quality visualization of data. PyNGL stands for 'Python Interface to the NCL Graphics Libraries,' and it is pronounced 'pingle.'",
         url = "http://www.pyngl.ucar.edu/",
         package_dir = { paf_pkg_name : ''},
         data_files  = DATA_FILES,
         ext_package = paf_pkg_name,
         ext_modules = EXT_MODULES
      )

#
# Clean up. Remove version file.
#
if os.path.exists(paf_vfile):
  os.system("/bin/rm -rf " + paf_vfile)
