#!/usr/bin/env python
#
# To install PyNGL:
#
#     python setup.py install
#

import os
import site
from distutils.core import setup, Extension

#
# Set from_src to True only if you are building PyNGL
# and all of its supplemental files from source code.
#
from_src = False

#
# Get the root of where PyNGL will live, and the extra PyNGL
# data files (fontcaps, graphcaps, map databases, example
# scripts, etc)
#
pkgs_pth   = site.sitedirs[0]
pyngl_dir  = pkgs_pth + "/PyNGL/ncarg"

if(from_src):
  ncl_root = os.getenv("NCARG_ROOT") + "/"
  ncl_lib  = ncl_root + "lib/"
else:
  ncl_root = ""
  ncl_lib  = ""

#
# List all the extra files that need to be installed with PyNGL.
# These files include example PyNGL scripts, data for the scripts,
# fonts, map databases, colormaps, and other databases.
#
asc_files      = os.listdir(ncl_lib + "ncarg/data/asc")
dbin_files     = os.listdir(ncl_lib + "ncarg/data/bin")
cdf_files      = os.listdir(ncl_lib + "ncarg/data/cdf")
grb_files      = os.listdir(ncl_lib + "ncarg/data/grb")
colormap_files = os.listdir(ncl_lib + "ncarg/colormaps")
database_files = os.listdir(ncl_lib + "ncarg/database")
fontcap_files  = os.listdir(ncl_lib + "ncarg/fontcaps")
graphcap_files = os.listdir(ncl_lib + "ncarg/graphcaps")
pynglex_files  = os.listdir(ncl_lib + "ncarg/pynglex")
bin_files      = ["ctrans","pynglex","psplit"]

#
# os.listdir doesn't include the relative directory path...
#
for i in xrange(len(asc_files)):
  asc_files[i] = ncl_lib + "ncarg/data/asc/" + asc_files[i]

for i in xrange(len(dbin_files)):
  dbin_files[i] = ncl_lib + "ncarg/data/bin/" + dbin_files[i]

for i in xrange(len(cdf_files)):
  cdf_files[i] = ncl_lib + "ncarg/data/cdf/" + cdf_files[i]

for i in xrange(len(grb_files)):
  grb_files[i] = ncl_lib + "ncarg/data/grb/" + grb_files[i]

for i in xrange(len(colormap_files)):
  colormap_files[i] = ncl_lib + "ncarg/colormaps/" + colormap_files[i]

for i in xrange(len(database_files)):
  database_files[i] = ncl_lib + "ncarg/database/" + database_files[i]

for i in xrange(len(fontcap_files)):
  fontcap_files[i] = ncl_lib + "ncarg/fontcaps/" + fontcap_files[i]

for i in xrange(len(graphcap_files)):
  graphcap_files[i] = ncl_lib + "ncarg/graphcaps/" + graphcap_files[i]

for i in xrange(len(pynglex_files)):
  pynglex_files[i] = ncl_lib + "ncarg/pynglex/" + pynglex_files[i]

for i in xrange(len(bin_files)):
  bin_files[i] = ncl_root + "bin/" + bin_files[i]

res_file = [ncl_lib + "ncarg/sysresfile"]

setup (name = "PyNGL",
       version="0.1.1b3",
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
