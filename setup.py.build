#!/usr/bin/env python
#
# This script is for building and installing PyNGL. It is assumed
# that NCAR Graphics and NCL libraries and include files have been
# built and installed to $NCARG_ROOT/{lib,include} and that
# applications like "ctrans" and "med" are in $NCARG_ROOT/bin.
#
# To build and install PyNGL:
#
#   python setup.build.py install
#
# To build PyNGL but not install it:
#
#   python setup.build.py build_ext
#
# To create a binary distribution:
#
#   python setup.build.py bdist_dumb --relative
#
# If no environment variables are set, this script will create
# a NumPy version of PyNGL.
#
# There are four environment variables that if set, will change
# the behavior of this script:
#
#    USE_NUMERIC   - Create a Numeric version of PyNGL only.
#
#    USE_NUMERPY   - Create a Numeric *and* NumPy version of PyNGL. This
#                    will create two packages: PyNGL and PyNGL_numeric.
#
#    USE_CVS         Use CVS to get the latest version of the pynglex files.
#
#    INCLUDE_PYNIO - Copy over PyNIO files from PyNIO installed location.
#                    and include as part of PyNGL package.
#
import sys,os,re
import fileinput
import shutil
from distutils.core import setup, Extension

#
# Determine whether we want to build a Numeric and/or NumPy version
# of PyNGL.  If the environment variable USE_NUMERIC is set, it will
# try to build a NumPy version. USE_NUMERIC doesn't need to be set to
# any value; it just has to be set.  If USE_NUMERPY is set, then
# both versions of PyNGL will be built, and the NumPy version will
# be put in package PyNGL, and the Numeric version in package PyNGL_numeric.
#
# HAS_NUM will be set by this script depending on USE_NUMERIC and USE_NUMERPY.
#
# HAS_NUM = 3 --> install both NumPy and Numeric versions of PyNGL
# HAS_NUM = 2 --> install NumPy version of PyNGL
# HAS_NUM = 1 --> install Numeric version of PyNGL
# HAS_NUM = 0 --> You're hosed as NumPy or Numeric don't exist.
#
try:
  path = os.environ["USE_NUMERPY"]
  HAS_NUM = 3
except:
  try:
    path = os.environ["USE_NUMERIC"]
    HAS_NUM = 1
  except:
    HAS_NUM = 2

#
# Test to make sure we actually the Numeric and/or NumPy modules
# that we have requested.
#
if HAS_NUM > 1:
  try:
    import numpy
  except ImportError:
    print "Cannot find NumPy; we'll try Numeric."
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
  print "Cannot find Numeric or NumPy; good-bye!"
  sys.exit()

#
# Should we copy over the PyNIO files and include them as part of
# the PyNGL distribution?  If so, you better have already run:
#
# python setup.py install
#
# in the pynio source directory. This script expects to find PyNIO
# installed in the PyNIO package if you are doing a NumPy build,
# and in the PyNIO_numeric package if you are doing a Numeric build.
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
# Should we copy all the installed files to somewhere under this
# directory so we can tar them up for a distribution?
#
try:
  path = os.environ["CREATE_DISTRIBUTION"]
  create_distribution = True
except:
  create_distribution = False
   
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
#  python setup.build.py build_ext -L/sw/lib
#
# You will then have to type "python setup.build.py install" separately to
# install the package.
#
ncl_and_sys_lib_paths = [ncl_lib, "/usr/X11R6/lib"]

if sys.platform == "darwin":
    dirs = ['/sw/lib','/Users/haley/lib/gcc-lib/i386-apple-darwin8.6.1/4.0.3']
    for dir in dirs:
      if(os.path.exists(dir)):
        ncl_and_sys_lib_paths.append(dir)

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
ncarg_dirs  = ["colormaps","data","database","fontcaps","graphcaps", \
               "grib2_codetables"]

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
# instead of this setup.build.py script.
#

if sys.platform == "irix6-64":
    print "Warning: This setup.build.py file will not work on an irix6-64 system."
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

#----------------------------------------------------------------------
#
# Loop through the modules for which we want to create versions of PyNGL.
#
#----------------------------------------------------------------------

for array_module in array_modules:
#----------------------------------------------------------------------
#
# Initialize variables for whether we are doing a Numeric or NumPy build.
# Some of these variables will be used as build (compilation) parameters.
#
#----------------------------------------------------------------------
  INCLUDE_PATHS = [ncl_inc]

  if array_module == 'Numeric':
    from Numeric import  __version__ as array_module_version

    pyngl_pkg_name = 'PyNGL_numeric'
    pynio_pkg_name = 'PyNIO_numeric'
    pyngl_pth_file = []     # No *.pth file for Numeric package, b/c we
                            # have to explicitly import it with 
                            # "import PyNGL_numeric.Ngl as Ngl" anyway.

    DMACROS =  [('NeedFuncProto',None)]

  else:
    from numpy import __version__ as array_module_version

    pyngl_pkg_name = 'PyNGL'
    pynio_pkg_name = 'PyNIO'
    pyngl_pth_file = [pyngl_pkg_name + '.pth']

#
# For a NumPy build, we need to point to the correct array "arrayobject.h"
# and set the USE_NUMPY macro for compiling the *.c files.
#
    INCLUDE_PATHS.insert(0,os.path.join(pkgs_pth,"numpy/core/include"))
    DMACROS =  [('NeedFuncProto', None),('USE_NUMPY',None)]

#----------------------------------------------------------------------
#
# Here are the instructions for compiling the "_hlu.so" file.
#
#----------------------------------------------------------------------
  print '====> Installing the',array_module,'version of PyNGL to the "'+pyngl_pkg_name+'" site packages directory.'

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
# "Walk" through the "ncarg" directories that we want to be a part of
# the distribution, and keep a list of them for later.
#
  pyngl_ncarg_files = [(pyngl_ncarg_dir, ['sysresfile'])]
  cwd = os.getcwd()
  os.chdir(ncl_ncarg_dir)
  for ncarg_dir in ncarg_dirs:
    for root, dirs, files in os.walk(ncarg_dir):
      for name in files:
        pyngl_ncarg_files.append((os.path.join(pyngl_ncarg_dir,root), \
                                  [os.path.join(ncl_ncarg_dir,root,name)]))
  os.chdir(cwd)
#
# If INCLUDE_PYNIO is set, then make sure we include the PyNIO files
# in the list of files to be packaged up with PyNGL.
#
  if include_pynio:
    print '====> Will be including the PyNIO files in the',pyngl_pkg_name,'package directory.'
    if sys.platform == "cygwin":
      pynio_files = ['Nio.py', 'pynio_version.py', 'nio.dll']
    else:
      pynio_files = ['Nio.py', 'pynio_version.py', 'nio.so']
    for i in xrange(len(pynio_files)):
      pynio_files[i] = os.path.join(pynio_dir,pynio_files[i])
  else:
    pynio_files = []

#
# This seems kludgy to me, but I need to make sure that if both 
# NumPy and Numeric versions of PyNGL are being built, we clean
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
# Numeric or NumPy specific operations.
#
  if array_module == 'Numeric':
    vfile.write("HAS_NUM = 1\n")
  else:
    vfile.write("HAS_NUM = 2\n")

  vfile.write("array_module_version = '%s'\n" % array_module_version)
  vfile.write("python_version = '%s'\n" % sys.version[:3])
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
  for i in xrange(len(pynglex_files)):
    pynglex_files[i] = os.path.join(pynglex_dir,pynglex_files[i])

#
# If we are doing a Numeric build, then we need to modify some of the
# pynglex example scripts.
#
  if array_module == 'Numeric':
    from mod_pynglex_files import *
    modify_pynglex_files(pynglex_files)

#----------------------------------------------------------------------
#
# Set up list of files we want to package up as part of the PyNGL
# package.
# 
#----------------------------------------------------------------------
  DATA_FILES = pyngl_ncarg_files
  DATA_FILES.append((os.path.join(pyngl_ncarg_dir,'pynglex'),pynglex_files))
  DATA_FILES.append((pkgs_pth, pyngl_pth_file))
  DATA_FILES.append((python_bin_dir,bin_files))
  DATA_FILES.append((pyngl_dir,py_files))
  DATA_FILES.append((pyngl_dir,pynio_files))
#
# Here's the setup function that will build and install everything.
#
  setup (name = pyngl_pkg_name,
         version = pyngl_version,
         author = "Dave Brown, Fred Clare, and Mary Haley",
         maintainer = "Mary Haley",
         maintainer_email = "haley@ucar.edu",
         description = "2D visualization library",
       long_description = "PyNGL is a Python language module designed for publication-quality visualization of data. PyNGL stands for 'Python Interface to the NCL Graphics Libraries,' and it is pronounced 'pingle.' It now contains the 'Nio' module, which enables NetCDF-like access for NetCDF (rw), HDF (rw), GRIB (r), and CCM (r) data files",
         url = "http://www.pyngl.ucar.edu/",
         package_dir = { pyngl_pkg_name : ''},
         data_files  = DATA_FILES,
         ext_package = pyngl_pkg_name,
         ext_modules = EXT_MODULES
      )

# 
# This section is for gathering up all the files we need to make 
# a complete PyNGL/PyNIO distribution that can be installed by
# an outside user with "python setup.py install".
#
  if create_distribution:
#
# Copy installed package files back to our current directory so we can
# "sdist" them into a distribution.
#
# First, remove some directories that we want to create from scratch.
#
    os.system('/bin/rm -rf ' + pyngl_pkg_name)
    os.system('/bin/rm -rf bin')
    os.system('/bin/rm -rf ' + pyngl_pkg_name + '-' + pyngl_version)
    os.mkdir(pyngl_pkg_name)
    os.mkdir('bin')

    os.system('/bin/cp -r ' + pyngl_dir + '/* ' + pyngl_pkg_name + '/.')
  
    for i in xrange(len(bin_files)):
      os.system('/bin/cp ' + bin_files[i] + ' bin/.')
#
# Copy over the appropriate "setup.py" file.
#
    os.system('/bin/cp setup.' + array_module + '.py setup.py')
#
# Modify it and update the version number in the file.
#
    for line in fileinput.input('setup.py',inplace=1):
      if (re.search("version=XXXX,",line) != None):
        print "version = '" + pyngl_version + "',"
      else:
        print line,

#
# Generate a MANIFEST.in file. This is the file that setup.py sdist
# uses to determine which files to tar up for the "source" distribution.
# In our case, because we don't release source code, the "source" 
# distribution is really a binary distribution.
#
    man_file = 'MANIFEST.in'
    if os.path.exists(man_file):
      os.system("/bin/rm -rf " + man_file)
    if os.path.exists('MANIFEST'):
      os.system("/bin/rm -rf MANIFEST")

    mfile = open(man_file,'w')
    if array_module == 'numpy':
      mfile.write("include PyNGL.pth\n")

    mfile.write("include setup.py\n")
    mfile.write("include README\n")
    mfile.write("recursive-include " + pyngl_pkg_name + " *\n")
    mfile.write("recursive-include bin *\n")
    mfile.close()

#
# Run the command to create the "source" distribution. This file
# name will not have a system name as part of the name.
#
    os.system("python setup.py sdist")

#
# Cleanup: remove the Scripts directory and pyngl_version.py file.
# If create_distribution was True, then remove files created by this process.
#

os.system("/bin/rm -rf " + pynglex_dir)

if os.path.exists(pyngl_vfile):
  os.system("/bin/rm -rf " + pyngl_vfile)


if create_distribution:
  print 'removing some files'
  print 'removing ' + pyngl_pkg_name
  print 'removing bin'
  print 'removing setup.py MANIFEST MANIFEST.in'
  os.system('/bin/rm -rf ' + pyngl_pkg_name)
  os.system('/bin/rm -rf bin')
  os.system('/bin/rm -rf setup.py MANIFEST MANIFEST.in')
