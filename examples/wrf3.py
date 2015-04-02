#
#  File:
#    wrf3.py
#
#  Synopsis:
#    Draws contours of a "TC" diagnostic variable calculated 
#    from WRF output file
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
import numpy, Nio, Ngl, os, sys


filename = "wrfout_d03_2012-04-22_23_00_00"
if(not os.path.exists(filename)):
  print "You do not have the necessary file to run this example."
  print "See the comments at the top of this script for more information."
  sys.exit()

#---Read data
a   = Nio.open_file(filename+".nc")  # Must add ".nc" suffix for Nio.open_file
T  = a.variables["T"][:]
P  = a.variables["P"][:]
PB = a.variables["PB"][:]

T  = T + 300
P  = P + PB
TC = Ngl.wrf_tk(P, T) - 273.16      # Convert to degC

#---Open file for graphics
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"wrf3")

#---Set some plot options
res                   = Ngl.Resources()

#---Contour options
res.cnFillOn          = True          # turn on contour fill
res.cnLinesOn         = False         # turn off contour lines
res.cnLineLabelsOn    = False         # turn off line labels
res.cnFillMode        = "RasterFill"        # These two resources
res.trGridType        = "TriangularMesh"    # can speed up plotting.
res.cnFillPalette     = "ncl_default"

res.lbOrientation     = "horizontal"   # default is vertical

res.tiMainString      = filename + ": temperature (degC)"
res.tiMainFontHeightF = 0.015

#
# Map options added using map info on WRF output file.
# This can make plotting go faster, because you don't have
# to do a map transformation.
#
res = Ngl.wrf_map_resources(a,res)

nt = 0       # first time step
nl = 0       # first level
plot = Ngl.contour_map(wks,TC[nt,nl,:,:],res)


Ngl.end()

