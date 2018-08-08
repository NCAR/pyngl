#
#  File:
#    wrf3.py
#
#  Synopsis:
#    Draws contours of a "TC" diagnostic variable calculated from a
#    from WRF output file using wrf-python
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
#    This example shows how to calculate the "TC" diagnostic from a 
#    WRF output file and draw filled contours.
#
#    If you have a large grid, then see the "cnFillMode" and
#    "trGridType" resource comments below.
#
#  Effects illustrated:
#    o  Using wrf-python to get data from WRF output file
#    o  Plotting WRF data in its native projection
#    o  Plotting curvilinear data
#    o  Using RasterFill for faster contouring
#    o  Using cnFillPalette to assign a color palette to contours
# 
#  Output:
#    This example produces a filled raster contour plot
#     
#  Notes:
#     You will need to include your own WRF output file in place
#     of the one referenced by this example.
#======================================================================
from __future__ import print_function
import numpy, Nio, Ngl, os, sys
from wrf import getvar, get_pyngl

filename = "wrfout_d03_2012-04-22_23_00_00"
if(not os.path.exists(filename)):
  print("You do not have the necessary '{}' file to run this example.".format(filename))
  print("You need to supply your own WRF output file")
  print("WRF output files usually have names like '{}'".format(filename))
  sys.exit()

#---Read data
a  = Nio.open_file(filename+".nc")  # Must add ".nc" suffix for Nio.open_file
tc = getvar(a,"tc")

#---Open file for graphics
wks_type = "png"
wks = Ngl.open_wks(wks_type,"wrf3")

# Set some map options based on information in WRF output file
res = get_pyngl(tc)
res.tfDoNDCOverlay    = True          # required for native projection

#---Contour options
res.cnFillOn          = True          # turn on contour fill
res.cnLinesOn         = False         # turn off contour lines
res.cnLineLabelsOn    = False         # turn off line labels
res.cnFillMode        = "RasterFill"        # These two resources
res.trGridType        = "TriangularMesh"    # can speed up plotting.
res.cnFillPalette     = "ncl_default"

res.lbOrientation      = "horizontal"   # default is vertical
res.pmLabelBarHeightF  = 0.08
res.pmLabelBarWidthF   = 0.65
res.lbTitleString      = "%s (%s)" % (tc.description,tc.units)
res.lbTitleFontHeightF = 0.015
res.lbLabelFontHeightF = 0.015

res.tiMainString      = filename
res.tiMainFont        = "helvetica-bold"
res.tiMainFontHeightF = 0.02

plot = Ngl.contour_map(wks,tc[0,:,:],res)

Ngl.end()

