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
#    This example shows how to read and unstagger U,V data on a WRF
#    output grid, and then draw streamlines. The streamlines are
#    dense, so the arrays are strided to cull some of them.
#
#  Effects illustrated:
#    o  Plotting WRF data in a lat/lon projection
#    o  Unstaggering WRF data
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

import numpy as np
import Nio, Ngl, os, sys

#----------------------------------------------------------------------
# This function takes a WRF variable and unstaggers it along the
# given dimension.
#----------------------------------------------------------------------
def wrf_unstagger(x):
  rank = len(x.shape)
  if rank < 2:
    print("wrf_unstagger: variable must be at least 2-dimensional")
    return x    

  xdims = x.dimensions
  if xdims[rank-1].endswith("_stag"):
    dim = "lon"
  elif xdims[rank-2].endswith("_stag"):
    dim = "lat"
  else:
    print("wrf_unstagger: error: couldn't find the staggered dimension")
    return x    

  if rank == 4:
    if dim == "lon":
      xu = 0.5*(x[:,:,:,:-1] + x[:,:,:,1:])
    else:
      xu = 0.5*(x[:,:,:-1,:] + x[:,:,1:,:])
  elif rank == 3:
    if dim == "lon":
      xu = 0.5*(x[:,:,:-1] + x[:,:,1:])
    else:
      xu = 0.5*(x[:,:-1,:] + x[:,1:,:])
  elif rank == 2:
    if dim == "lon":
      xu = 0.5*(x[:,:-1] + x[:,1:])
    else:
      xu = 0.5*(x[:-1,:] + x[1:,:])
  return xu

# Read data
filename = "wrfout_d03_2012-04-22_23_00_00"
if(not os.path.exists(filename)):
  print("You do not have the necessary file to run this example.")
  print("You need to supply your own WRF output file")
  print("WRF output files usually have names like '%s'" % filename)
  sys.exit()

# Read some WRF data
a    = Nio.open_file(filename+".nc")  # Must add ".nc" suffix for Nio.open_file
u    = a.variables["U"]
v    = a.variables["V"]
latu = a.variables["XLAT_U"]
lonu = a.variables["XLONG_U"]

# Unstagger the data
ua  = wrf_unstagger(u)
va  = wrf_unstagger(v)
lat = wrf_unstagger(latu)
lon = wrf_unstagger(lonu)

# First timestep, lowest (bottommost) level, every 5th lat/lon
nl    = 0
nt    = 0
nstep = 5     # a stride to cull some of the streamlines
u10   = ua[nt,nl,::nstep,::nstep]
v10   = va[nt,nl,::nstep,::nstep]
spd   = np.sqrt(u10**2+v10**2)                

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
res.mpLandFillColor         = "beige"
res.mpOceanFillColor        = "transparent"
res.mpInlandWaterFillColor  = "transparent"
res.mpGridLatSpacingF       = 1
res.mpGridLonSpacingF       = 1
#res.mpGridAndLimbOn        = False
 
res.stLineThicknessF  = 3.0          # Thicker streamlines
res.stMonoLineColor   = False        # Use multiple colors for streamlines

res.tiMainString      = "U10/V10 streamlines color by wind speed"

# Necessary to overlay on map correctly.
res.vfYArray          = lat[0,::nstep,::nstep]
res.vfXArray          = lon[0,::nstep,::nstep]

plot = Ngl.streamline_scalar_map(wks,u10,v10,spd,res)

Ngl.end()

