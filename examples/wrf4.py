#
#  File:
#    wrf4.py
#
#  Synopsis:
#    Draws colored streamlines of U,V read off a WRF output file. 
#
#  Categories:
#    Streamlines
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    April, 2015
#
#  Description:
#    This example shows how to read U,V data on a WRF
#    output grid using wrf-python. The streamlines are
#    dense, so the arrays are strided to cull some of them.
#
#  Effects illustrated:
#    o  Using wrf-python to get U,V data from WRF output file
#    o  Drawing streamlines
#    o  Coloring streamlines by another field
# 
#  Output:
#    This example produces a colored streamline plot
#     
#  Notes:
#     You will need to include your own WRF output file in place
#     of the one referenced by this example.
#======================================================================

from __future__ import print_function
import numpy as np
import Nio, Ngl, os, sys
from wrf import getvar, latlon_coords, to_np


# Read data
filename = "wrfout_d03_2012-04-22_23_00_00"
if(not os.path.exists(filename)):
  print("You do not have the necessary '{}' file to run this example.".format(filename))
  print("You need to supply your own WRF output file")
  print("WRF output files usually have names like '{}'".format(filename))
  sys.exit()

# Read some WRF data
a  = Nio.open_file(filename+".nc")  # Must add ".nc" suffix for Nio.open_file
ua = getvar(a,"ua")
va = getvar(a,"va")

# First timestep, lowest (bottommost) level, every 5th lat/lon
nl    = 0
nt    = 0
nstep = 5     # a stride to cull some of the streamlines
u     = ua[nl,::nstep,::nstep]
v     = va[nl,::nstep,::nstep]
spd   = np.sqrt(u**2+v**2)

# Get the latitude and longitude points
lat, lon = latlon_coords(ua)
lat = to_np(lat)
lon = to_np(lon)

# Open file for graphics
wks_type = "png"
wks = Ngl.open_wks(wks_type,"wrf4")

res                   = Ngl.Resources()

res.mpDataBaseVersion = "MediumRes"         # Better map outlines
res.mpLimitMode       = "LatLon"            # Zoom in on map area of interest
res.mpMinLatF         = np.min(lat[:])-0.1
res.mpMaxLatF         = np.max(lat[:])+0.1
res.mpMinLonF         = np.min(lon[:])-0.1
res.mpMaxLonF         = np.max(lon[:])+0.1

res.mpFillOn                = True
res.mpLandFillColor         = "gray85"
res.mpOceanFillColor        = "transparent"
res.mpInlandWaterFillColor  = "transparent"
res.mpGridLatSpacingF       = 1
res.mpGridLonSpacingF       = 1
#res.mpGridAndLimbOn        = False
 
res.stLineThicknessF  = 3.0          # Thicker streamlines
res.stMonoLineColor   = False        # Use multiple colors for streamlines

res.tiMainString      = "U10/V10 streamlines color by wind speed"

# Necessary to overlay on map correctly.
res.vfYArray          = lat[::nstep,::nstep]
res.vfXArray          = lon[::nstep,::nstep]

plot = Ngl.streamline_scalar_map(wks,u,v,spd,res)

Ngl.end()

