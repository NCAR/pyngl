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
# Get the root of where PyNGL will live.
#
pyngl_dir = site.sitedirs[0] + "/PyNGL"

#
# List out all the extra files. There has to be a way to
# automate this, so we don't need to list every file. We
# need something like "os.listdir", but we need it to
# return the directory path in front of the filename.
# i.e. "pynglex/chkbay.py" and not just "chkbay.py"
#
asc_files      = os.listdir('data/asc')
cdf_files      = os.listdir('data/cdf')
colormap_files = os.listdir('colormaps')
database_files = os.listdir('database')
fontcap_files  = os.listdir('fontcaps')
graphcap_files = os.listdir('graphcaps')
pynglex_files  = os.listdir('pynglex')

for i in xrange(len(asc_files)):
  asc_files[i] = "data/asc/" + asc_files[i]

for i in xrange(len(cdf_files)):
  cdf_files[i] = "data/cdf/" + cdf_files[i]

for i in xrange(len(colormap_files)):
  colormap_files[i] = "colormaps/" + colormap_files[i]

for i in xrange(len(database_files)):
  database_files[i] = "database/" + database_files[i]

for i in xrange(len(fontcap_files)):
  fontcap_files[i] = "fontcaps/" + fontcap_files[i]

for i in xrange(len(graphcap_files)):
  graphcap_files[i] = "graphcaps/" + graphcap_files[i]

for i in xrange(len(pynglex_files)):
  pynglex_files[i] = "pynglex/" + pynglex_files[i]

pynglex = ['pynglex']

res_file = ['sysresfile']

setup (name = "PyNGL",
       version="0.1.0b1",
       author="Fred Clare and Mary Haley",
       author_email="fred@ucar.edu,haley@ucar.edu",
       description = "2D visualization library",
       long_description = "PyNGL is a Python language module designed for publication-quality visualization of data. PyNGL stands for 'Python Interface to the NCL Graphics Libraries,' and it is pronounced 'pingle.'",
       url = "http://ngwww.ucar.edu/ncl/pyngl/",
       packages = ['PyNGL'],
       extra_path = 'PyNGL',
       ext_modules = [Extension('PyNGL._hlu',[])],
       data_files = [("bin",                   ['bin/pynglex']),
                     (pyngl_dir,               ['sysresfile']),
                     (pyngl_dir + "/data/asc", asc_files),
                     (pyngl_dir + "/data/cdf", cdf_files),
                     (pyngl_dir + "/colormaps",colormap_files),
                     (pyngl_dir + "/database", database_files),
                     (pyngl_dir + "/fontcaps", fontcap_files),
                     (pyngl_dir + "/graphcaps",graphcap_files),
                     (pyngl_dir + "/pynglex",  pynglex_files)]
      )
