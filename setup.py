#!/usr/bin/env python
#
# This script is for building and installing PyNGL. It is assumed
# that NCAR Graphics and NCL libraries and include files have been
# built and installed to $NCARG_ROOT/{lib,include} and that
# applications like "ctrans" and "med" are in $NCARG_ROOT/bin.
#
# To build and install PyNGL:
#
#   python setup.py install
#
# To build PyNGL but not install it:
#
#   python setup.py build_ext
#
# To create a binary distribution:
#
#   python setup.py bdist_dumb --relative
#
# If no environment variables are set, this script will create
# a Numeric version of PyNGL.
#
# There are four environment variables that if set, will change
# the behavior of this script:
#
#    USE_NUMPY     - Create a numpy version of PyNGL only.
#
#    USE_NUMERPY   - Create a Numeric *and* numpy version of PyNGL. This
#                    will create two packages: PyNGL and PyNGL_numpy.
#    USE_CVS         Use CVS to get the latest version of the pynglex files.
#
#    INCLUDE_PYNIO - Copy over PyNIO files from PyNIO installed location.
#                    and include as part of PyNGL package.
#
import sys,os
import shutil
from distutils.core import setup, Extension

#
# Determine whether we want to build a Numeric and/or Numpy version
# of PyNGL.  If the environment variable USE_NUMPY is set, it will
# try to build a numpy version. USE_NUMPY doesn't need to be set to
# any value; it just has to be set.  If USE_NUMERPY is set, then
# both versions of PyNGL will be built, and the Numeric version will
# be put in package PyNGL, and the numpy version in package PyNGL_numpy.
#
# HAS_NUM will be set by this script depending on USE_NUMPY and USE_NUMERPY.
#
# HAS_NUM = 3 --> install both numpy and Numeric versions of PyNGL
# HAS_NUM = 2 --> install numpy version of PyNGL
# HAS_NUM = 1 --> install Numeric version of PyNGL
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
# Should we copy over the PyNIO files and include them as part of
# the PyNGL distribution?  If so, you better have already run:
#
# python setup.py install
#
# in the pynio source directory. This script expects to find PyNIO
# installed in the PyNIO package if you are doing a Numeric build,
# and in the PyNIO_numpy package if you are doing a numpy build.
#
# This script does not yet check if PyNIO is installed on your system,
# so it will complain if it can't find the files.
#
try:
  path = os.environ["INCLUDE_PYNIO"]
  include_pynio = True
except:
  include_pynio = False

#
# Should we use CVS for getting the latest pynglex examples?
#
try:
  path = os.environ["USE_CVS"]
  use_cvs = True
except:
  use_cvs = False
   
#
# Initialize some variables.
#
pyngl_vfile     = "pyngl_version.py"         # PyNGL version file.
pkgs_pth        = os.path.join(sys.prefix, 'lib', 'python'+sys.version[:3],
                               'site-packages')
python_bin_dir  = os.path.join(sys.prefix,'bin')
pynglex_dir     = "Scripts"                  # Don't change this!

#
# Get root and various other directories of installed files.
#
ncl_root      = os.getenv("NCARG_ROOT")
ncl_bin       = os.path.join(ncl_root,'bin')
ncl_lib       = os.path.join(ncl_root,'lib')
ncl_inc       = os.path.join(ncl_root,'include')
ncl_ncarg_dir = os.path.join(ncl_lib,'ncarg')
ncl_data_dir  = os.path.join(ncl_ncarg_dir,'data')

#
# Gather up the executables we want to install as part of PyNGL.
# We will get the NCAR Graphics executables from the installed
# location ($NCARG_ROOT/bin).
#
bin_files = ["ctrans","med","psplit"]
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
ncl_and_sys_lib_paths = [ncl_lib, "/usr/X11R6/lib"]

if sys.platform == "darwin":
    ncl_and_sys_lib_paths.append('/sw/lib')

if sys.platform == "linux2" and os.uname()[-1] == "x86_64":
    ncl_and_sys_lib_paths.append('/usr/X11R6/lib64')

#----------------------------------------------------------------------
#
# List some of the extra files that need to be installed with PyNGL 
# (some other files will be listed inside the "do" loop).
#
# These files include data files, fonts, map databases, colormaps,
# and other databases.
#
#----------------------------------------------------------------------
#
# "os.listdir" doesn't include the relative directory path.
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
if os.path.exists(os.path.join(ncl_ncarg_dir,'database','rangs')):
  database_files.remove("rangs")

#
# os.listdir doesn't include the relative directory path, so add it
# back here. There's gotta be a better way to do this.
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

res_file = ['sysresfile']

#
# Gather up the *.py module files.
#
py_files= ['Ngl.py','hlu.py','__init__.py','pyngl_version.py']

#----------------------------------------------------------------------
#
# Start setting up some build parameters
#
#----------------------------------------------------------------------
#
# List the extra arguments and libraries that we need on the load line.
#
EXTRA_LINK_ARGS = ""
LIBRARIES = ["nfpfort", "hlu", "ncarg", "ncarg_gks", "ncarg_c", "ngmath", "X11", "g2c"]

#
# The IRIX system is problematic, because distuils uses "-all" as one of the
# options to "ld".  This causes all objects from all archives to be linked
# in, and hence you get some undefined references from libraries like the
# Spherepack library, which you shouldn't need (yet). The solution around this
# is to use "-notall" in conjunction with "-all", but the "extra_link_args"
# parameter that you are supposed to use puts this at the *end* of the ld
# line, which doesn't work. It needs to be at the beginning.
#
# So, for now, on tempest, I'm having to build the darn *.so file
# by hand with:
#
# ld -64 -shared -all build/temp.irix64-6.5-2.4/Helper.o \
#   build/temp.irix64-6.5-2.4/hlu_wrap.o build/temp.irix64-6.5-2.4/gsun.o \
#  -L/fis/scd/home/ncargd/dev/opt/IRIX64_6.5_mips4_64/lib -L/usr/X11R6/lib \
#  -lnfpfort -lhlu -lncarg -lncarg_gks -lncarg_c -lngmath -lX11 -lftn -lm \
#  -o build/lib.irix64-6.5-2.4/PyNGL/_hlu.so -notall
#
# I use the build_on_irix64 script for this. Run the build_on_irix64 script
# instead of this setup.py script.
#

if sys.platform == "irix6-64":
    print "Warning: This setup.py file will not work on an irix6-64 system."
    print "Use 'build_on_irix64' instead."
#
# This is for later, if we ever get this to work under IRIX.
#
    LIBRARIES.remove('g2c')
    LIBRARIES.append('ftn')
    LIBRARIES.append('m')
    EXTRA_LINK_ARGS = ['-notall']

if sys.platform == "aix5":
    LIBRARIES.remove('g2c')
    LIBRARIES.append('xlf90')

#----------------------------------------------------------------------
#
# Loop through the modules for which we want to create versions of PyNGL.
#
#----------------------------------------------------------------------

for array_module in array_modules:
#----------------------------------------------------------------------
#
# Initialize variables for whether we are doing a Numeric or numpy build.
# Some of these variables will be used as build (compilation) parameters.
#
#----------------------------------------------------------------------
  INCLUDE_PATHS = [ncl_inc]

  if array_module == 'Numeric':
    from Numeric import  __version__ as array_module_version

    pyngl_pkg_name = 'PyNGL'
    pynio_pkg_name = 'PyNIO'
    pyngl_pth_file = [pyngl_pkg_name + '.pth']

    DMACROS =  [('NeedFuncProto',None)]

  else:
    from numpy import __version__ as array_module_version

    pyngl_pkg_name = 'PyNGL_numpy'
    pynio_pkg_name = 'PyNIO_numpy'
    pyngl_pth_file = []             # No *.pth file for numpy package!

#
# For a numpy build, we need to point to the correct array "arrayobject.h"
# and set the USE_NUMPY macro for compiling the *.c files.
#
    INCLUDE_PATHS.insert(0,os.path.join(pkgs_pth,"numpy/core/include"))
    DMACROS =  [('NeedFuncProto', None),('USE_NUMPY',None)]

#----------------------------------------------------------------------
#
# Here are the instructions for compiling the "_hlu.so" file.
#
#----------------------------------------------------------------------
  print '====> Installing the',array_module,'version of PyNGL to the',pyngl_pkg_name,'package directory.'

  EXT_MODULES = [Extension('_hlu', 
                 ['Helper.c','hlu_wrap.c','gsun.c'],
                  define_macros   = DMACROS,
                  extra_link_args = EXTRA_LINK_ARGS,
                  include_dirs    = INCLUDE_PATHS,
                  library_dirs    = ncl_and_sys_lib_paths,
                  libraries       = LIBRARIES)]

#----------------------------------------------------------------------
#
# Get the directories of where the extra PyNGL data files (fontcaps,
# graphcaps, map databases, example scripts, etc) will be installed.
#
#----------------------------------------------------------------------
  pyngl_dir       = os.path.join(pkgs_pth, pyngl_pkg_name)
  pynio_dir       = os.path.join(pkgs_pth, pynio_pkg_name)

  pyngl_ncarg_dir = os.path.join(pyngl_dir, 'ncarg')
  pyngl_data_dir  = os.path.join(pyngl_ncarg_dir, 'data')

#
# If INCLUDE_PYNIO is set, then make sure we include the PyNIO files
# in the list of files to be packaged up with PyNGL.
#
  if include_pynio:
    print '====> Will be including the PyNIO files in the',pyngl_pkg_name,'package directory.'
    pynio_files = ['Nio.py', 'pynio_version.py', 'nio.so']
    for i in xrange(len(pynio_files)):
      pynio_files[i] = os.path.join(pynio_dir,pynio_files[i])
  else:
    pynio_files = []

#
# This seems kludgy to me, but I need to make sure that if both 
# numpy and Numeric versions of PyNGL are being built, we clean
# the *.o files beforehand. This is because "setup" puts the *.o files
# in the same location (build/temp.xxxx/.) every time, regardless of which
# package we're building. Maybe there's a way to tell setup to put the
# *.o files in a different directory, but I haven't found it yet.
#
  if len(array_modules) > 1:
    print "====> Removing build's *.o and *.so files..."
    os.system("find build -name '*.o' -exec /bin/rm {} \;")
    os.system("find build -name '*.so' -exec /bin/rm {} \;")

#----------------------------------------------------------------------
#
# Create version file that contains version and array module info.
#
#----------------------------------------------------------------------
  if os.path.exists(pyngl_vfile):
    os.system("/bin/rm -rf " + pyngl_vfile)

  pyngl_version = open('version','r').readlines()[0].strip('\n')

  vfile = open(pyngl_vfile,'w')
  vfile.write("version = '%s'\n" % pyngl_version)
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

#----------------------------------------------------------------------
#
# Get the list of pynglex *.py and *.res files. You have a choice of
# checking out a new directory from CVS, or using the "examples" directory.
# If you use the "examples" directory, make sure it doesn't have 
# any extraneous files.
#
#----------------------------------------------------------------------
  os.system("/bin/rm -rf " + pynglex_dir)

  if use_cvs:
    os.system("cvs co pynglex")
    pynglex_files = os.listdir(pynglex_dir)
#
# Remove everything but *.py and *.res files from the list of files.
#
    pynglex_files.remove("yMakefile")
    pynglex_files.remove("CVS")
    pynglex_files.remove("pynglex")
  else:
#
# Create a Scripts directory and copy the .py and .res files
# from the ../examples directory into the Scripts directory.
# The executable ../examples/pynglex must also be copied.
#
    all_pynglex_files = os.listdir("../examples")
    os.system("mkdir " + pynglex_dir)
    pynglex_files = []
    for file in all_pynglex_files:
      if (file[-3:] == ".py" or file[-4:] == ".res"):
        pynglex_files.append(file)
        os.system("cp ../examples/" + file + " " + pynglex_dir)
    os.system("cp ../examples/pynglex " + pynglex_dir)
#
# Prepend the full directory path leading to files.
#
  pynglex_numpy_files = []
  for i in xrange(len(pynglex_files)):
    pynglex_files[i] = os.path.join(pynglex_dir,pynglex_files[i])

#
# If we are doing a numpy build, then we need to modify some of the
# pynglex example scripts.
#
  if array_module == 'numpy':
    from mod_pynglex_files import *
    modify_pynglex_files(pynglex_files)

#----------------------------------------------------------------------
#
# Set up list of files we want to package up as part of the PyNGL
# package.
# 
#----------------------------------------------------------------------
  DATA_FILES = [(os.path.join(pyngl_ncarg_dir,'pynglex'),pynglex_files),
                (pkgs_pth, pyngl_pth_file),
                (python_bin_dir,bin_files),
                (pyngl_dir,py_files),
                (pyngl_dir,pynio_files),
                (os.path.join(pyngl_data_dir,'asc'), asc_files),
                (os.path.join(pyngl_data_dir,'bin'), dbin_files),
                (os.path.join(pyngl_data_dir,'cdf'), cdf_files),
                (os.path.join(pyngl_data_dir,'grb'), grb_files),
                (os.path.join(pyngl_ncarg_dir,'colormaps'),colormap_files),
                (os.path.join(pyngl_ncarg_dir,'database'), database_files),
                (os.path.join(pyngl_ncarg_dir,'fontcaps'), fontcap_files),
                (os.path.join(pyngl_ncarg_dir,'graphcaps'),graphcap_files),
                (pyngl_ncarg_dir, res_file)]
#
# Here's the setup function that will build and install everything.
#
  setup (name = pyngl_pkg_name,
         version = pyngl_version,
         author = "Fred Clare and Mary Haley",
         maintainer = "Mary Haley",
         maintainer_email = "haley@ucar.edu",
         description = "2D visualization library",
         long_description = "PyNGL is a Python language module designed for publication-quality visualization of data. PyNGL stands for 'Python Interface to the NCL Graphics Libraries,' and it is pronounced 'pingle.'",
         url = "http://www.pyngl.ucar.edu/",
         package_dir = { pyngl_pkg_name : ''},
         data_files  = DATA_FILES,
         ext_package = pyngl_pkg_name,
         ext_modules = EXT_MODULES
      )

#
# Cleanup: remove the Scripts directory and pyngl_version.py file.
#
os.system("/bin/rm -rf " + pynglex_dir)

if os.path.exists(pyngl_vfile):
  os.system("/bin/rm -rf " + pyngl_vfile)
