#!/usr/bin/env python
#
# This script is for building and installing PyNGL.
#
# To install, type:
# 
#  python setup.py install
#
# To build PyNGL from source, you must have the NCL/NCAR Graphics
# software installed on your system.
#
# The NCARG_ROOT environment variable must be set to the root
# directory of where NCL/NCAR Graphics software was installed.
#
# See http://www.ncl.ucar.edu/Download/ for information on 
# installing NCL/NCAR Graphics (available as one package).
#

import sys, os, re, platform, fileinput
from distutils.core import setup, Extension
from distutils.util import get_platform
from distutils.sysconfig import get_python_lib

# Create file containing PyNGL and numpy version.
def create_version_file():
  if os.path.exists(pyngl_vfile):
    os.remove(pyngl_vfile)

  pyngl_version = open('version','r').readlines()[0].strip('\n')

  vfile = open(pyngl_vfile,'w')
  vfile.write("version = '%s'\n" % pyngl_version)
  vfile.write("array_module = 'numpy'\n")
  vfile.write("array_module_version = '%s'\n" % array_module_version)
  vfile.write("python_version = '%s'\n" % sys.version[:3])
  vfile.close()

# Copy the pynglex script to same filename w/the python version appended.
def copy_pynglex_script():
  pynglex_v_file = os.path.join(pynglex_dir,"pynglex"+sys.version[:3])
  os.system("cp " + os.path.join(pynglex_dir,"pynglex") + " " + pynglex_v_file)

# Modify the pynglex script to have the correct python invocation.
  for line in fileinput.input(pynglex_v_file,inplace=1):
    if (re.search("/usr/bin/env python",line) != None):
      print line.replace("python","python"+sys.version[:3]),
    elif(re.search("^py_cmd = 'python'",line) != None):
      print line.replace("python","python"+sys.version[:3]),
    else:
      print line,

# Return list of pynglex examples and resource files.
def get_pynglex_files():
  all_pynglex_files = os.listdir(pynglex_dir)

  pynglex_files = []
  for file in all_pynglex_files:
    if (file[-3:] == ".py" or file[-4:] == ".res"):
      pynglex_files.append(os.path.join(pynglex_dir,file))
  return pynglex_files

# Return list of files we need under $NCARG_ROOT/lib/ncarg.
def get_ncarg_files():
  ncl_lib       = os.path.join(ncl_root,'lib')
  ncl_ncarg_dir = os.path.join(ncl_lib,'ncarg')
  ncarg_dirs    = ["colormaps","data","database","fontcaps","graphcaps"]

  cwd = os.getcwd()          # Retain current directory.
  os.chdir(ncl_ncarg_dir)    # CD to $NCARG_ROOT/lib/ncarg

  pyngl_ncarg_files = []
# Walk through each directory and gather up files. Skip over
# the rangs directory.
  for ncarg_dir in ncarg_dirs:
    for root, dirs, files in os.walk(ncarg_dir):
      names = []
      for name in files:
        if root != "database/rangs":
          names.append(os.path.join(ncl_ncarg_dir,root,name))
      if names != []:
        pyngl_ncarg_files.append((os.path.join(pyngl_ncarg_dir,root),names))

  os.chdir(cwd)    # CD

  return pyngl_ncarg_files

# Return list of executables we need under $NCARG_ROOT/bin.
def get_bin_files():
  ncl_bin         = os.path.join(ncl_root,'bin')
  bin_files       = ["ctrans","med","psplit"]
  pyngl_bin_files = []

  for file in bin_files:
    if(os.path.exists(os.path.join(ncl_bin,file))):
      pyngl_bin_files.append(os.path.join(ncl_bin,file))

# Add the "pynglex" script and the "pynglex.2.x" python version script.
  pyngl_bin_files.append(os.path.join(pynglex_dir,'pynglex'))
  pyngl_bin_files.append(os.path.join(pynglex_dir,'pynglex'+sys.version[:3]))

  return pyngl_bin_files

# Return list of libraries and paths needed for compilation
def set_ncl_libs_and_paths():
  ncl_lib = os.path.join(ncl_root,'lib')

  PATHS = [ncl_lib, "/usr/X11R6/lib"]

  if sys.platform == "darwin":
    dir = '/sw/lib'
    if(os.path.exists(dir)):
      PATHS.append(dir)

  if sys.platform == "linux2" and os.uname()[-1] == "x86_64":
    dir = '/usr/X11R6/lib64'
    if(os.path.exists(dir)):
      PATHS.append(dir)

# Libraries needed to compile _hlu.so/fplib.so modules.
  LIBS = ["nfpfort", "hlu", "ncarg", "ncarg_gks", "ncarg_c",
          "ngmath", "png", "z", "X11"]

# Add extra library needed for C/Fortran interfacing.
  if sys.platform == "linux2" and os.uname()[-1] == "x86_64" and \
    platform.python_compiler() == "GCC 4.1.1":
    LIBS.append("gfortran")
  elif sys.platform == "aix5":
    LIBS.append("xlf90")
  elif sys.platform == "darwin" and "i386" in os.uname():
      LIBS.append('f95')
  elif sys.platform == "sunos5":
      os.environ["CC"]="/opt/SUNWspro/bin/cc"
      LIBS.append('f77compat')
      LIBS.append('fsu')
      LIBS.append('sunmath')
  else:
    LIBS.append("g2c")

  return LIBS,PATHS

# Return list of include paths needed for compilation
def set_include_paths():
  ncl_inc = os.path.join(ncl_root,'include')
  PATHS = [ncl_inc]

# Location of numpy's "arrayobject.h".
  PATHS.insert(0,os.path.join(pkgs_pth,"numpy/core/include"))

  return PATHS

#----------------------------------------------------------------------
# Main section
#----------------------------------------------------------------------

long_description = "PyNGL is a Python language module designed for publication-quality visualization of data. PyNGL stands for 'Python Interface to the NCL Graphics Libraries,' and it is pronounced 'pingle.' It now contains the 'Nio' module, which enables NetCDF-like access for NetCDF (rw), HDF (rw), GRIB (r), and CCM (r) data files"

# I read somewhere that distutils doesn't update this file properly
# when the contents of directories change.

if os.path.exists('MANIFEST'): os.remove('MANIFEST')

PYNGL_PKG_NAME = 'PyNGL'                    # Name of package to install.
pyngl_pth_file = [PYNGL_PKG_NAME + '.pth']  # and it's *.pth file.
pkgs_pth       = get_python_lib()

#
# Make sure NCARG_ROOT is set. This is only necessary if building
# PyNGL from source code.

ncl_root = os.getenv("NCARG_ROOT")
if ncl_root == None:
  print "Error: NCARG_ROOT environment variable not set. Can't continue."
  sys.exit()

# Construct the version file.
try:
  import numpy
except ImportError:
  print "Error: Cannot import NumPy. Can't continue."
  sys.exit()
from numpy import __version__ as array_module_version

pyngl_vfile = "pyngl_version.py"     # Name of version file.
create_version_file()

# Get directories of installed NCL/NCAR Graphics libraries and include files

LIBRARIES,LIB_PATHS = set_ncl_libs_and_paths()
INC_PATHS           = set_include_paths()

# Set some compile options.
if os.uname()[-1] == "x86_64" or \
  (os.uname()[-1] == "Power Macintosh" and os.uname()[2] == "7.9.0"):
  os.environ["CFLAGS"] = "-O2"
DMACROS =  [('NeedFuncProto', None)]

# Instructions for compiling the "_hlu.so" and "fplib.so" files.
EXT_MODULES = [Extension('_hlu', 
              ['Helper.c','hlu_wrap.c','gsun.c'],
                define_macros   = DMACROS,
                include_dirs    = INC_PATHS,
                library_dirs    = LIB_PATHS,
                libraries       = LIBRARIES),
               Extension('fplib', 
               [os.path.join('paft','fplibmodule.c')],
                define_macros   = DMACROS,
                include_dirs    = INC_PATHS,
                library_dirs    = LIB_PATHS,
                libraries       = LIBRARIES)]

# Set the directories of where the extra PyNGL data files (fontcaps,
# graphcaps, map databases, example scripts, etc) will be installed.
pyngl_dir       = os.path.join(pkgs_pth, PYNGL_PKG_NAME)
pyngl_ncarg_dir = os.path.join(pyngl_dir, 'ncarg')
pyngl_data_dir  = os.path.join(pyngl_ncarg_dir, 'data')
pynglex_dir     = "examples"
python_bin_dir  = os.path.join(sys.prefix,'bin')

pynglex_files = get_pynglex_files()   # Get example files associated
                                      # with the "pynglex" script.

copy_pynglex_script()  # Copy pynglex script to itself with
                       # python version appended

# Create list of supplemental files needed.
DATA_FILES = get_ncarg_files()          # We need NCARG_ROOT for the lib
DATA_FILES.append((python_bin_dir,get_bin_files()))      # and bin files.
DATA_FILES.append((pkgs_pth, pyngl_pth_file))
DATA_FILES.append((pyngl_ncarg_dir, ['sysresfile']))
DATA_FILES.append((os.path.join(pyngl_ncarg_dir,'pynglex'),pynglex_files))

setup (name = 'PyNGL',
       version          = '1.3.0',
       author           = 'Dave Brown, Fred Clare, and Mary Haley',
       author_email     = 'dbrown@ucar.edu, haley@ucar.edu',
       maintainer       = 'Mary Haley',
       maintainer_email = 'haley@ucar.edu',
       description      = '2D visualization library',
       long_description = long_description,
       url              = 'http://www.pyngl.ucar.edu/',
       package_dir      = { PYNGL_PKG_NAME : ''},
       packages         = [PYNGL_PKG_NAME],
       data_files       = DATA_FILES,
       ext_package      = PYNGL_PKG_NAME,
       ext_modules      = EXT_MODULES
    )

