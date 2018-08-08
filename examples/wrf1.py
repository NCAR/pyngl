#
#  File:
#    wrf1.py
#
#  Synopsis:
#    Draws filled contours over a map of a variable or diagnostic calculated from a WRF file.
#
#  Categories:
#    Contouring
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    April 2015
#
#  Description:
#    This example shows how to use wrf-python to read a variable or
#    calculate a diagnostic from a WRF output file and draw filled
#    contours over a map. See wrf2.py for a similar script that draws
#    the data in the native WRF projection with shapefile outlines added.
#
#  Effects illustrated:
#    o  Using wrf-python to get data from WRF output file
#    o  Plotting WRF data
#    o  Plotting curvilinear data
#    o  Using cnFillPalette to assign a color palette to contours
#    o  Explicitly defining contour levels.
# 
#  Output:
#    This example produces a filled contour plot
#     
#  Notes:
#     You will need to include your own WRF output file in place
#     of the one referenced by this example.
#======================================================================
from __future__ import print_function
import numpy, Nio, Ngl, os, sys
from wrf import getvar, latlon_coords, to_np

filename = "wrfout_d01_2005-12-14_13:00:00"
if(not os.path.exists(filename)):
  print("You do not have the necessary '{}' file to run this example.".format(filename))
  print("You need to supply your own WRF output file")
  print("WRF output files usually have names like '{}'".format(filename))
  sys.exit()

#---Read data
f        = Nio.open_file("{}.nc".format(filename))  # Must add ".nc" suffix for Nio.open_file
var_name = "slp"           # "slp", "ter"
var      = getvar(f,var_name)
lat, lon = latlon_coords(var)

lat = to_np(lat)
lon = to_np(lon)

#print(max(var.XLAT))

#---Open file for graphics
wks_type = "png"
wks = Ngl.open_wks(wks_type,"wrf1")

# Create handle for plot options
res = Ngl.Resources()

# Contour options
res.cnFillOn          = True          # turn on contour fill
res.cnLinesOn         = False         # turn off contour lines
res.cnLineLabelsOn    = False         # turn off line labels
res.sfXArray          = lon
res.sfYArray          = lat

# This is for terrain
if var_name == "ter":
  res.cnFillPalette        = "OceanLakeLandSnow"
  res.cnLevelSelectionMode = "ExplicitLevels"
  res.cnLevels  = [2,50,75,100,150,200,250,300,350,400,500,600,700,800,900,1000,1100]
elif var_name == "slp":
  res.cnFillPalette     = "MPL_viridis"

# Map options
res.mpDataBaseVersion     = "MediumRes"                # better map outlines
res.mpOutlineBoundarySets = "GeophysicalAndUSStates"   # more outlines

res.mpLimitMode           = "LatLon"
res.mpMinLatF             = numpy.min(lat)-1
res.mpMaxLatF             = numpy.max(lat)+1
res.mpMinLonF             = numpy.min(lon)-1
res.mpMaxLonF             = numpy.max(lon)+1
res.mpGridAndLimbOn       = False

# Labelbar options
res.lbOrientation      = "horizontal"
res.lbLabelFontHeightF = 0.01
res.pmLabelBarHeightF  = 0.08
res.pmLabelBarWidthF   = 0.65
res.lbTitleString      = "%s (%s)" % (var.description,var.units)
res.lbTitleFontHeightF = 0.01

# Main Title
res.tiMainString      = "WRF curvilinear lat/lon grid (" + str(var.shape[0]) + " x " + str(var.shape[1]) + ")"
res.tiMainFontHeightF = 0.02

plot = Ngl.contour_map(wks,var,res)

Ngl.end()
