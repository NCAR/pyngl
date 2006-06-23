#!/usr/bin/env python
#
# To build and/or install PyNGL:
#
#   python setup.py install
#
# There are four environment variables, that if set, will change
# the behavior of this script:
#
#    USE_NUMERPY   - Create a Numeric *and* numpy version of PyNGL. This
#                    will create two packages: PyNGL and PyNG_numpy.
#    USE_NUMPY     - Create a numpy version of PyNGL
#    USE_CVS         Use CVS to get the latest version of the pynglex files.
#    INCLUDE_PYNIO - Copy over PyNIO files from PyNIO installed location.
#                    and include as part of PyNGL package.
#
import sys,os
import shutil
import fileinput
import re
import tempfile

from distutils.core import setup, Extension

#
# Determine whether we want to build a Numeric and/or Numpy version
# of PyNGL.  If the environment variable USE_NUMPY is set, it will
# try to build a NumPy version. USE_NUMPY doesn't need to be set to
# any value; it just has to be set.  If USE_NUMERPY is set, then
# both versions of PyNGL will be built, and the Numeric version will
# be put in package PyNGL, and the numpy version in package PyNGL_numpy.
#
# HAS_NUM will be set by this script depending on USE_NUMPY and USE_NUMERPY.
#
# HAS_NUM = 3 --> install both numpy and Numeric versions of module
# HAS_NUM = 2 --> install numpy version of module
# HAS_NUM = 1 --> install Numeric version of module
# HAS_NUM = 0 --> You're hosed, you have neither module
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
    try:
      print "Cannot find numpy; we'll try Numeric."
      HAS_NUM = 1
    except ImportError:
      print "Cannot find Numeric or numpy; good-bye!"
      exit

if HAS_NUM == 1 or HAS_NUM == 3:
  try:
    import Numeric
  except ImportError:
    HAS_NUM = HAS_NUM-1
    if HAS_NUM == 0:
      print "Cannot find Numeric or numpy; good-bye!"
      exit

#
# Should we copy over the PyNIO files and include them as part of
# the PyNGL distribution?
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
# Create pyngl_version.py file that contains version and
# array module info.
#
pyngl_vfile = "pyngl_version.py"
os.system("/bin/rm -rf " + pyngl_vfile)

pyngl_version = open('version','r').readlines()[0].strip('\n')

vfile = open(pyngl_vfile,'w')
vfile.write("version = '%s'\n" % pyngl_version)

if HAS_NUM == 2:
  vfile.write("HAS_NUM = 2\n")
  from numpy import __version__ as array_module_version
  vfile.write("array_module = 'numpy'\n")
else:
  vfile.write("HAS_NUM = 1\n")
  from Numeric import  __version__ as array_module_version
  vfile.write("array_module = 'Numeric'\n")

vfile.write("array_module_version = '%s'\n" % array_module_version)
vfile.close()

#
# Get the root of where PyNGL will live, and where the extra PyNGL
# data files (fontcaps, graphcaps, map databases, example
# scripts, etc) will be installed. Note that for the most part, these
# files will be shared if both Numeric and numpy versions are being
# installed. The only thing that is not shared are pynglex example
# scripts.
#
pkgs_pth        = os.path.join(sys.prefix, 'lib', 'python'+sys.version[:3],
                               'site-packages')
python_bin_dir  = os.path.join(sys.prefix,'bin')
if HAS_NUM == 1 or HAS_NUM == 3:
  pyngl_dir       = os.path.join(pkgs_pth, os.path.join('PyNGL'))
  pynio_dir       = os.path.join(pkgs_pth, os.path.join('PyNIO'))
elif HAS_NUM == 2:
  pyngl_dir       = os.path.join(pkgs_pth, os.path.join('PyNGL_numpy'))
  pynio_dir       = os.path.join(pkgs_pth, os.path.join('PyNIO_numpy'))

pyngl_ncarg_dir = os.path.join(pyngl_dir, os.path.join('ncarg'))
pyngl_data_dir  = os.path.join(pyngl_ncarg_dir, 'data')
#
# Set PyNGL_numpy directory paths if we are doing two packages.
#
if HAS_NUM == 3:
  pyngl_numpy_dir       = os.path.join(pkgs_pth, os.path.join('PyNGL_numpy'))
  pynio_numpy_dir       = os.path.join(pkgs_pth, os.path.join('PyNIO_numpy'))
  pyngl_numpy_ncarg_dir = os.path.join(pyngl_numpy_dir, os.path.join('ncarg'))
  pyngl_numpy_data_dir  = os.path.join(pyngl_numpy_ncarg_dir, 'data')

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
# any extraneous files.
#
# If HAS_NUM is 3, then we need to check out this directory two times:
# one for Numeric and one for numpy.
#
pynglex_dir = "Scripts"                  # Don't change this!
os.system("/bin/rm -rf " + pynglex_dir)
if HAS_NUM == 3:
  pynglex_numpy_dir = pynglex_dir + "_numpy"
  os.system("/bin/rm -rf " + pynglex_numpy_dir)

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
# Make copy of files so we can modify them for a numpy version.
#
if HAS_NUM == 3:
  os.system("cp -r " + pynglex_dir + " " + pynglex_numpy_dir)

#
# Prepend the full directory path leading to files.
#
pynglex_numpy_files = []
for i in xrange(len(pynglex_files)):
  if HAS_NUM == 3:
    pynglex_numpy_files.append(os.path.join(pynglex_numpy_dir,pynglex_files[i]))
  pynglex_files[i] = os.path.join(pynglex_dir,pynglex_files[i])

####################################################################
#                                                                  #
#  Begin code for mods to the example sources for NumPy support.   #
#                                                                  #
####################################################################
#
# Modify the example sources appropriately if NumPy support is
# requested.  For all examples except "metrogram.py," "scatter1.py," 
# and "ngl09p.py" this is just a matter of replacing "import Numeric"
# with "import numpy as Numeric".  The cases of "meteogram.py"
# and "ngl09p.py" are handled as special cases in the if block below; 
# "scatter1.py" is then handled separately.
#
if HAS_NUM == 2:
  pynglex_files_to_mod = pynglex_files
elif HAS_NUM == 3:
  pynglex_files_to_mod = pynglex_numpy_files

if HAS_NUM > 1:
  for line in fileinput.input(pynglex_files_to_mod,inplace=1):
    if (re.search("import Numeric",line) != None):
      print "import numpy as Numeric"
    elif(re.search("^import Ngl",line) != None):
      print "import PyNGL_numpy.Ngl as Ngl"
    elif(re.search("^import Nio",line) != None):
      print "import PyNGL_numpy.Nio as Nio"
    elif (os.path.basename(fileinput.filename()) == "meteogram.py" and  \
        re.search("typecode()",line) != None):
      print line.replace("typecode()","dtype.char"),
    elif (os.path.basename(fileinput.filename()) == "ngl09p.py" and     \
        re.search("import MA",line) != None):
      print line.replace("import MA","import numpy.core.ma as MA"),
    elif (os.path.basename(fileinput.filename()) == "ngl09p.py" and     \
        re.search("MA.Float0",line) != None):
      print line.replace("MA.Float0","dtype=float"),
    else:
      print line,
  for file in pynglex_files_to_mod:
    if (os.path.basename(file) == "scatter1.py"):
      scatter_src = open(file,"r")
      scatter_new = tempfile.TemporaryFile()

      while(1):
        line = scatter_src.readline()
        if (line == ""):
          break
        elif (re.search("From Scientific import",line) != None):
          while (re.search("^from",line) == None):
            line = scatter_src.readline()
          line = scatter_src.readline()
        elif (re.search("Put the data",line) != None):
          while (re.search("^plot =",line) == None):
            line = scatter_src.readline()
          line = scatter_src.readline()
          scatter_new.write("""#
#  Do a quadratic least squares fit.
#
npoints = len(x)
a = Numeric.zeros([npoints,3],Numeric.Float32)
for m in xrange(npoints):
  a[m,0] = 1.
  for j in xrange(1,3):
    a[m,j] = x[m]*a[m,j-1]
c = (Numeric.linalg.lstsq(a,y,rcond=1.e-15))[0]

#
#  Draw the least squares quadratic curve.
#
num  = 301
delx = 1000./num
xp    = Numeric.zeros(num,Numeric.Float0)
yp    = Numeric.zeros(num,Numeric.Float0)
for i in xrange(num):
  xp[i] = float(i)*delx
  yp[i] = c[0]+c[1]*xp[i]+c[2]*xp[i]*xp[i]
plot = Ngl.xy(wks,xp,yp,resources) # Draw least squares quadratic.

""")
        scatter_new.write(line)

#
#  Write the new NumPy source back over the Numeric source.
#
      scatter_src.close()
      scatter_src = open(file,"w+")
      scatter_new.seek(0)
      for line in scatter_new.readlines():
        scatter_src.write(line)
      scatter_src.close()
      scatter_new.close()
########################################################
#                                                      #
#  End of mods to example sources for NumPy support.   #
#                                                      #
########################################################

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
ncl_and_sys_lib_paths = [ncl_lib, "/usr/X11R6/lib"]

if sys.platform == "darwin":
    ncl_and_sys_lib_paths.append('/sw/lib')

if sys.platform == "linux2" and os.uname()[-1] == "x86_64":
    ncl_and_sys_lib_paths.append('/usr/X11R6/lib64')


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
if os.path.exists(os.path.join(ncl_ncarg_dir,'database','rangs')):
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
# If INCLUDE_PYNIO is set, then make sure we include the PyNIO files.
#

pynio_files = []
if HAS_NUM == 3:
  pynio_numpy_files = []
if include_pynio:
  pynio_files = ['Nio.py', 'pynio_version.py', 'nio.so']
  for i in xrange(len(pynio_files)):
    if HAS_NUM == 3:
      pynio_numpy_files.append(pynio_files[i])
    pynio_files[i] = os.path.join(pynio_dir,pynio_files[i])
    if HAS_NUM == 3:
      pynio_numpy_files[i] = os.path.join(pynio_numpy_dir,pynio_numpy_files[i])

#
# List the extra arguments and libraries that we need on the load line.
#
EXTRA_LINK_ARGS = ""
LIBRARIES = ["nfpfort", "hlu", "ncarg", "ncarg_gks", "ncarg_c", "ngmath", "X11", "g2c"]

INCLUDE_PATHS = ncl_inc

if HAS_NUM == 2:
  INCLUDE_PATHS.insert(0,os.path.join(pkgs_pth,"numpy/core/include"))

if HAS_NUM == 3:
  INCLUDE_NUMPY_PATHS = [os.path.join(pkgs_pth,"numpy/core/include"),
                         os.path.join(ncl_root,'include')]

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

if HAS_NUM == 2:
  print '====> building with numpy/arrayobject.h'
  DMACROS =  [('NeedFuncProto', None),('USE_NUMPY',None)]
elif HAS_NUM == 1:
  print '====> building with Numeric/arrayobject.h'
  DMACROS =  [('NeedFuncProto',None)]
else:
  print '====> building with Numeric and numpy arrayobject.h'
  DMACROS      =  [('NeedFuncProto',None)]
  DNUMPYMACROS =  [('NeedFuncProto', None),('USE_NUMPY',None)]

EXT_MODULES = [Extension('_hlu', 
              ['Helper.c','hlu_wrap.c','gsun.c'],
               define_macros = DMACROS,
               extra_link_args = EXTRA_LINK_ARGS,
               include_dirs = INCLUDE_PATHS,
               library_dirs = ncl_and_sys_lib_paths,
               libraries = LIBRARIES)]

DATA_FILES = [(os.path.join(pyngl_ncarg_dir,'pynglex'),pynglex_files),
              (pkgs_pth,                ["PyNGL.pth"]),
              (python_bin_dir,bin_files),
              (pyngl_dir,py_files),
              (os.path.join(pyngl_data_dir,'asc'), asc_files),
              (os.path.join(pyngl_data_dir,'bin'), dbin_files),
              (os.path.join(pyngl_data_dir,'cdf'), cdf_files),
              (os.path.join(pyngl_data_dir,'grb'), grb_files),
              (os.path.join(pyngl_ncarg_dir,'colormaps'),colormap_files),
              (os.path.join(pyngl_ncarg_dir,'database'), database_files),
              (os.path.join(pyngl_ncarg_dir,'fontcaps'), fontcap_files),
              (os.path.join(pyngl_ncarg_dir,'graphcaps'),graphcap_files),
              (pyngl_dir,pynio_files),
              (pyngl_ncarg_dir, [res_file])]
#
# Here's the setup function.
#
if HAS_NUM == 1 or HAS_NUM == 3:
  setup (name = "PyNGL",
       version = pyngl_version,
       author = "Fred Clare and Mary Haley",
       maintainer = "Mary Haley",
       maintainer_email = "haley@ucar.edu",
       description = "2D visualization library",
       long_description = "PyNGL is a Python language module designed for publication-quality visualization of data. PyNGL stands for 'Python Interface to the NCL Graphics Libraries,' and it is pronounced 'pingle.'",
       url = "http://www.pyngl.ucar.edu/",
       package_dir = { 'PyNGL' : ''},
       data_files = DATA_FILES,
       ext_package = 'PyNGL',
       ext_modules = EXT_MODULES
      )
elif HAS_NUM == 2:
  setup (name = "PyNGL_numpy",
       version = pyngl_version,
       author = "Fred Clare and Mary Haley",
       maintainer = "Mary Haley",
       maintainer_email = "haley@ucar.edu",
       description = "2D visualization library",
       long_description = "PyNGL is a Python language module designed for publication-quality visualization of data. PyNGL stands for 'Python Interface to the NCL Graphics Libraries,' and it is pronounced 'pingle.'",
       url = "http://www.pyngl.ucar.edu/",
       package_dir = { 'PyNGL_numpy' : ''},
       data_files = DATA_FILES,
       ext_package = 'PyNGL_numpy',
       ext_modules = EXT_MODULES
      )

#
# if HAS_NUM is 3, then this means we just created a Numeric
# version of PyNGL, and now we need to create a numpy version.
#
if HAS_NUM == 3:
#
# Create a new pyngl_version.py file that contains version and
# array module info for numpy.
#
  os.system("/bin/rm -rf " + pyngl_vfile)

  vfile = open(pyngl_vfile,'w')
  vfile.write("version = '%s'\n" % pyngl_version)
  vfile.write("HAS_NUM = 2\n")
  from numpy import __version__ as array_module_version
  vfile.write("array_module = 'numpy'\n")
  vfile.write("array_module_version = '%s'\n" % array_module_version)
  vfile.close()

#
# Start with fresh build.
#
  os.system("find build -name '*.o' -exec /bin/rm {} \;")

  DATA_FILES = [(os.path.join(pyngl_numpy_ncarg_dir,'pynglex'),
                pynglex_numpy_files),
                (os.path.join(pkgs_pth,'PyNGL_numpy'), py_files),
                (os.path.join(pyngl_numpy_data_dir,'asc'), asc_files),
                (os.path.join(pyngl_numpy_data_dir,'bin'), dbin_files),
                (os.path.join(pyngl_numpy_data_dir,'cdf'), cdf_files),
                (os.path.join(pyngl_numpy_data_dir,'grb'), grb_files),
                (os.path.join(pyngl_numpy_ncarg_dir,'colormaps'),colormap_files),
                (os.path.join(pyngl_numpy_ncarg_dir,'database'), database_files),
                (os.path.join(pyngl_numpy_ncarg_dir,'fontcaps'), fontcap_files),
                (os.path.join(pyngl_numpy_ncarg_dir,'graphcaps'),graphcap_files),
                (pyngl_numpy_dir,pynio_numpy_files),
                (pyngl_numpy_ncarg_dir, [res_file])]

  EXT_MODULES = [Extension('_hlu',
                 ['Helper.c','hlu_wrap.c','gsun.c'],
                 define_macros = DNUMPYMACROS,
                 extra_link_args = EXTRA_LINK_ARGS,
                 include_dirs = INCLUDE_NUMPY_PATHS,
                 library_dirs = ncl_and_sys_lib_paths,
                 libraries = LIBRARIES)]

  setup (name = "PyNGL_numpy",
         version = pyngl_version,
         author = "Fred Clare and Mary Haley",
         maintainer = "Mary Haley",
         maintainer_email = "haley@ucar.edu",
         description = "2D visualization library",
         long_description = "PyNGL is a Python language module designed for publication-quality visualization of data. PyNGL stands for 'Python Interface to the NCL Graphics Libraries,' and it is pronounced 'pingle.'",
         url = "http://www.pyngl.ucar.edu/",
         package_dir = { 'PyNGL_numpy' : ''},
         data_files = DATA_FILES,
         ext_package = 'PyNGL_numpy',
         ext_modules = EXT_MODULES
      )
  os.system("/bin/rm -rf " + pynglex_numpy_dir)


#
# Cleanup: remove the Scripts directory.
#
os.system("/bin/rm -rf " + pynglex_dir)
