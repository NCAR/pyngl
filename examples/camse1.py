#
#  File:
#    camse1.py
#
#  Synopsis:
#    Creates color contours of a CAM-SE variable over an
#    orthographic map.
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
#    This example shows how to draw smooth contours of an
#    unstructured CAM-SE grid.
#
#  Effects illustrated:
#    o  Plotting CAM-SE data
#    o  Plotting unstructured data
# 
#  Output:
#    This example produces a single contour plot.
#     
#  Notes:
#     The CAM-SE data file is too large to be included with this 
#     software. Either use your own CAM-SE grid, or send email to
#     pyngl-talk@ucar.edu.
#
#     http://mailman.ucar.edu/mailman/listinfo/pyngl-talk
#
from __future__ import print_function
import numpy, Nio, Ngl, sys, os

#---Read data
filename = "b.e12.B1850C5CN.ne30_g16.init.ch.027.cam.h0.0001-01.nc"
if(not os.path.exists(filename)):
  print("You do not have the necessary '{}' file to run this example.".format(filename))
  print("See the comments at the top of this script for more information.")
  sys.exit()

a     = Nio.open_file(filename)
vname = "TS"
data  = a.variables[vname]    
lat   = a.variables["lat"][:]        # 1D array (48602 cells)
lon   = a.variables["lon"][:]        # ditto

ncells = data.shape[1]
print("There are {} cells in the {} variable".format(ncells,vname))

wks_type = "png"
wks = Ngl.open_wks(wks_type,"camse1")

#---Set some plot options
res                   = Ngl.Resources()

# Contour options
res.cnFillOn          = True          # turn on contour fill
res.cnLinesOn         = False         # turn off contour lines
res.cnLineLabelsOn    = False         # turn off line labels
res.cnLevelSpacingF   = 2.5           # NCL chose 5.0
res.cnFillPalette     = "WhiteBlueGreenYellowRed"

# Map options
res.mpProjection      = "Orthographic"
res.mpCenterLonF      = 40
res.mpCenterLatF      = 60
res.mpPerimOn         = False

# Not sure why I need this.
res.pmTickMarkDisplayMode = "Never"

# Main Title
res.tiMainString      = "{} ({}) ({} cells)".format(data.long_name,
                                                    data.units,
                                                    ncells)
res.tiMainFontHeightF = 0.018

# Labelbar options
res.lbLabelFontHeightF = 0.01

#---Additional resources needed for putting contours on map
res.sfXArray          = lon
res.sfYArray          = lat

plot = Ngl.contour_map(wks,data[0,:],res)

Ngl.end()

