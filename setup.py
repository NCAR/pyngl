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
# Get the root of where PyNGL will live, and the extra PyNGL
# data files (fontcaps, graphcaps, map databases, example
# scripts, etc)
#
pkgs_pth  = site.sitedirs[0]
pyngl_dir = pkgs_pth + "/PyNGL/ncarg"

#
# List all the extra files that need to be installed with PyNGL.
# These files include example PyNGL scripts, data for the scripts,
# fonts, map databases, colormaps, and other databases.
#
asc_files      = os.listdir('ncarg/data/asc')
dbin_files     = os.listdir('ncarg/data/bin')
cdf_files      = os.listdir('ncarg/data/cdf')
grb_files      = os.listdir('ncarg/data/grb')
colormap_files = os.listdir('ncarg/colormaps')
database_files = os.listdir('ncarg/database')
fontcap_files  = os.listdir('ncarg/fontcaps')
graphcap_files = os.listdir('ncarg/graphcaps')
pynglex_files  = os.listdir('ncarg/pynglex')
bin_files      = os.listdir('bin')

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
       version="0.1.1b2",
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
