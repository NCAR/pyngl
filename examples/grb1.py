#
#  File:
#    grb1.py
#
#  Synopsis:
#    Plots GRIB2 data on a rotated grid.
#
#  Category:
#    Contours over maps
#    Maps
#
#  Author:
#    Mary Haley (based on NCL example from Dave Brown)
#  
#  Date of initial publication:
#    April, 2015
#
#  Description:
#
#  Effects illustrated:
#    o  Reading data from a GRIB2 file
#    o  Plotting GRIB2 data in its native rotated projection
#
#
#  Output:
#    A single visualization is produced showing scaled brightness 
#    temperature.

import numpy as np
import Nio, Ngl, os, sys

# Read data
dir  = Ngl.pynglpath("data")
flnm = "MET9_IR108_cosmode_0909210000.grb2"
print(os.path.join(dir,"grb",flnm))
a    = Nio.open_file(os.path.join(dir,"grb",flnm))
sbt  = a.variables["SBTMP_P31_GRLL0"]
lat  = a.variables["gridlat_0"]
lon  = a.variables["gridlon_0"]
nlat = lat.shape[0]
nlon = lat.shape[1]

wks_type = "png"
wks = Ngl.open_wks(wks_type,"grb1")

# Set some plot options
res                   = Ngl.Resources()

# Contour options
res.cnFillOn           = True          # turn on contour fill
res.cnLinesOn          = False         # turn off contour lines
res.cnLineLabelsOn     = False         # turn off line labels
res.cnFillPalette      = "BlGrYeOrReVi200"
res.lbLabelFontHeightF = 0.015              # default is a bit large

# Set resources necessary to get map projection correct.
res.mpLimitMode        = "Corners"
res.mpLeftCornerLatF   = lat[nlat-1][0] 
res.mpLeftCornerLonF   = lon[nlat-1][0]
res.mpRightCornerLatF  = lat[0][nlon-1]      
res.mpRightCornerLonF  = lon[0][nlon-1] 
res.mpCenterLonF       = lon.Longitude_of_southern_pole 
res.mpCenterLatF       = lon.Latitude_of_southern_pole + 90
res.trYReverse         = True
res.tfDoNDCOverlay     = True

# Set other map resources
res.mpGridAndLimbOn       = False
res.mpDataBaseVersion     = "MediumRes"
res.mpDataSetName         = "Earth..2"
res.mpOutlineBoundarySets = "AllBoundaries"

# Main Title
res.tiMainString      = sbt.long_name

plot = Ngl.contour_map(wks,sbt[:],res)

Ngl.end()

