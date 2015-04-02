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

filename = "wrfout_d01_2005-12-14_13:00:00"
if(not os.path.exists(filename)):
  print "You do not have the necessary file to run this example."
  sys.exit()
filename = filename + ".nc"        # must add the suffix for Nio.open_file

#---Read data
a   = Nio.open_file(filename)
hgt = a.variables["HGT"][0,:,:]     # Read first time step ( nlat x nlon)
lat = a.variables["XLAT"][0,:,:]    # 2D array (nlat x nlon)
lon = a.variables["XLONG"][0,:,:]   # ditto

#---Open PNG for graphics
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"wrf1")

#---Set some plot options
res                   = Ngl.Resources()

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

res.mpLimitMode           = "LatLon"
res.mpMinLatF             = numpy.min(lat)-1
res.mpMaxLatF             = numpy.max(lat)+1
res.mpMinLonF             = numpy.min(lon)-1
res.mpMaxLonF             = numpy.max(lon)+1
res.mpGridAndLimbOn       = False

# Labelbar options
res.lbOrientation      = "horizontal"
res.lbLabelFontHeightF = 0.01

# Main Title
dims = hgt.shape
res.tiMainString      = "WRF curvilinear lat/lon grid (" + str(dims[0]) + " x " + str(dims[1]) + ")"

# Additional resources needed for putting contours on map
res.sfXArray          = lon
res.sfYArray          = lat

plot = Ngl.contour_map(wks,hgt,res)

Ngl.end()

