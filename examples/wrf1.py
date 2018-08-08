#
#  File:
#    wrf1.py
#
#  Synopsis:
#    Draws contours over a map of a "HGT" variable read off a WRF output file.
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
#    This example shows how to read the height variable off a WRF output file 
#    and draw filled contours. See wrf2.py for a similar script that 
#    draws the height in the native projection with shapefile outlines added.
#
#  Effects illustrated:
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
import numpy, Nio, Ngl, os, sys
from wrf import getvar, get_pyngl

filename = "wrfout_d01_2005-12-14_13:00:00"
if(not os.path.exists(filename)):
  print("You do not have the necessary '%s' file to run this example." % filename)
  print("You need to supply your own WRF output file")
  print("WRF output files usually have names like '%s'" % filename)
  sys.exit()

#---Read data
a   = Nio.open_file(filename+".nc")  # Must add ".nc" suffix for Nio.open_file
var = getvar(a,"ter")

#---Open file for graphics
wks_type = "png"
wks = Ngl.open_wks(wks_type,"wrf1")

#---Set some plot options
res = get_pyngl(var)

# Contour options
res.cnFillOn          = True          # turn on contour fill
res.cnLinesOn         = False         # turn off contour lines
res.cnLineLabelsOn    = False         # turn off line labels
res.cnFillPalette     = "OceanLakeLandSnow"
res.cnLevelSelectionMode = "ExplicitLevels"
res.cnLevels  = [2,50,75,100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2200]

# Map options
res.mpDataBaseVersion     = "MediumRes"                # better map outlines
res.mpOutlineBoundarySets = "GeophysicalAndUSStates"   # more outlines
res.mpGridAndLimbOn       = False

# Labelbar options
res.lbOrientation      = "horizontal"
res.lbLabelFontHeightF = 0.01

# Allows us to overlay data in native WRF map projection
res.tfDoNDCOverlay     = True

# Main Title
res.tiMainString      = "WRF curvilinear lat/lon grid (" + str(var.shape[0]) + " x " + str(var.shape[1]) + ")"

plot = Ngl.contour_map(wks,var,res)

Ngl.end()

