#!/usr/bin/env python
#
# To install PyNGL:
#
#     python setup.py install
#

import os,sys
from os.path import join
import site
import shutil
from distutils.core import setup, Extension

#
# Set copy_files to True only if you first need to copy over
# PyNGL supplemental files.  You should only do this if you
# have built NCARG/NCL/PyNGL from source code.
#
# Set copy_rangs to True if you want to copy over the RANGS/GSHHS
# database. This database takes up about 100 megabytes.
#
# These variables are for internal use only.
#
copy_files = False
copy_rangs = False

#
# Get the root of where PyNGL will live, and the extra PyNGL
# data files (fontcaps, graphcaps, map databases, example
# scripts, etc)
#
pkgs_pth  = os.path.join(sys.prefix, 'lib', 'python'+sys.version[:3],
            'site-packages')
pyngl_dir = pkgs_pth + "/PyNGL/ncarg"

#
# List directories that we need to get supplemental files from.
#
main_dirs   = ["ncarg","bin"]
ncarg_dirs  = ["data","colormaps","database","fontcaps","graphcaps",\
               "pynglex"]
ncarg_files = ["sysresfile"]
bin_files   = ["ctrans","med","psplit","pynglex"]

if(copy_files):
#
# Get root directory of the files.
#
  ncl_root = os.getenv("NCARG_ROOT") + "/"
  ncl_lib  = ncl_root + "lib/"

  moved_rangs    = False
  rangs_dir_from = ncl_lib + "ncarg/database/rangs"
  rangs_dir_to   = ncl_lib + "ncarg/rangs"
  if(not copy_rangs and os.path.exists(rangs_dir_from)):
    print "Moving rangs database out of the way..."
    os.rename(rangs_dir_from,rangs_dir_to)
    moved_rangs = True
#
# Remove local directories if they exist, so we can start anew.
#
  for i in xrange(len(main_dirs)):
    shutil.rmtree(main_dirs[i],ignore_errors=True)
    os.mkdir(main_dirs[i])

#
# Copy over the files.
#
  for i in xrange(len(ncarg_dirs)):
    shutil.copytree(ncl_lib + "ncarg/" + ncarg_dirs[i],"ncarg/"+ncarg_dirs[i])
  for i in xrange(len(ncarg_files)):
    shutil.copy(ncl_lib + "ncarg/" + ncarg_files[i],"ncarg/")
  for i in xrange(len(bin_files)):
    shutil.copy(ncl_root + "bin/" + bin_files[i],"bin/")
#
# Copy rangs dir back, if necessary.
#
  if(moved_rangs):
    print "Moving rangs database back..."
    os.rename(rangs_dir_to,rangs_dir_from)

del bin_files

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

asc_files      = os.listdir("ncarg/data/asc")
dbin_files     = os.listdir("ncarg/data/bin")
cdf_files      = os.listdir("ncarg/data/cdf")
grb_files      = os.listdir("ncarg/data/grb")
colormap_files = os.listdir("ncarg/colormaps")
database_files = os.listdir("ncarg/database")
fontcap_files  = os.listdir("ncarg/fontcaps")
graphcap_files = os.listdir("ncarg/graphcaps")
pynglex_files  = os.listdir("ncarg/pynglex")
bin_files      = os.listdir("bin")

#
# os.listdir doesn't include the relative directory path...
#
for i in xrange(len(asc_files)):
  asc_files[i] = "ncarg/data/asc/" + asc_files[i]

for i in xrange(len(dbin_files)):
  dbin_files[i] = "ncarg/data/bin/" + dbin_files[i]

for i in xrange(len(cdf_files)):
  cdf_files[i] = "ncarg/data/cdf/" + cdf_files[i]

for i in xrange(len(grb_files)):
  grb_files[i] = "ncarg/data/grb/" + grb_files[i]

for i in xrange(len(colormap_files)):
  colormap_files[i] = "ncarg/colormaps/" + colormap_files[i]

for i in xrange(len(database_files)):
  database_files[i] = "ncarg/database/" + database_files[i]

for i in xrange(len(fontcap_files)):
  fontcap_files[i] = "ncarg/fontcaps/" + fontcap_files[i]

for i in xrange(len(graphcap_files)):
  graphcap_files[i] = "ncarg/graphcaps/" + graphcap_files[i]


for i in xrange(len(pynglex_files)):
  pynglex_files[i] = "ncarg/pynglex/" + pynglex_files[i]

for i in xrange(len(bin_files)):
  bin_files[i] = "bin/" + bin_files[i]

res_file = ["ncarg/sysresfile"]

setup (name = "PyNGL",
       version="0.1.1b6",
       author="Fred Clare and Mary Haley",
       author_email="fred@ucar.edu,haley@ucar.edu",
       description = "2D visualization library",
       long_description = "PyNGL is a Python language module designed for publication-quality visualization of data. PyNGL stands for 'Python Interface to the NCL Graphics Libraries,' and it is pronounced 'pingle.'",
       url = "http://www.pyngl.ucar.edu/",
       packages = ['PyNGL'],
       data_files = [("bin",                   bin_files),
                     (pkgs_pth,                ["PyNGL.pth"]),
                     (pkgs_pth+"/PyNGL",       ["PyNGL/_hlu.so"]),
                     (pyngl_dir,               res_file),
                     (pyngl_dir + "/data/asc", asc_files),
                     (pyngl_dir + "/data/bin", dbin_files),
                     (pyngl_dir + "/data/cdf", cdf_files),
                     (pyngl_dir + "/data/grb", grb_files),
                     (pyngl_dir + "/colormaps",colormap_files),
                     (pyngl_dir + "/database", database_files),
                     (pyngl_dir + "/fontcaps", fontcap_files),
                     (pyngl_dir + "/graphcaps",graphcap_files),
                     (pyngl_dir + "/pynglex",  pynglex_files)]
      )
