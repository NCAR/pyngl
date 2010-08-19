#
#  File:
#    panel2.py
#
#  Synopsis:
#    Illustrates how to draw three visualizations on a single frame.
#
#  Category:
#    Paneling
#    Contours over maps
#    Label bar
#
#  Authors:
#    Mary Haley and Fred Clare
#  
#  Date of initial publication:
#    November, 2006
#
#  Description:
#    Illustrates how to put multiple plots on a page and how
#    to have a common label bar for all plots on the page.
#    Also illustrates the three map database resolutions.
#
#  Effects illustrated:
#    o  How to handle missing values.
#    o  How to put multiple plots on a page.
#    o  Drawing contours over maps.
#    o  Some special paneling resources.
#    o  A common label bar for all plots.
#    o  The three map database resolutions.
# 
#  Output:
#    This examples produces a single visualization showing
#    three individual contour plots on a single frame with
#    a common label bar.
#
#  Notes:
#    The geographic area in the maps is the sea of Japan.
#    The data are artificially generated with a missing
#    value of 1.e+10 outside the contoured area.
#     
#    In order to get high-resolution coastlines,  you must have the 
#    RANGS/GSHHS database installed. Otherwise, this example will
#    revert to using medium-resolution coastlines.
#
#    For information on getting the RANGS/GSHHS database, see:
#
#        http://www.pyngl.ucar.edu/Graphics/rangs.shtml

import numpy
import os
import Ngl
import Nio

#
# Open the netCDF file containing the data.
#
dirc  = Ngl.pynglpath("data")
data_file = "panel2.nc"
file = Nio.open_file(os.path.join(dirc,"cdf","panel2.nc"),"r")
fsd = file.variables["FSD"][0,:,:]  # Read some data off file.
lat = file.variables["lat"][:]
lon = file.variables["lon"][:]

#
# Start graphics.
#
wks_type = "ps"
rlist            = Ngl.Resources()
rlist.wkColorMap = "BlGrYeOrReVi200"
wks = Ngl.open_wks(wks_type,"panel2",rlist)

igray = Ngl.new_color(wks,0.7,0.7,0.7)             # Add gray to colormap

res = Ngl.Resources()
res.nglDraw                = False     # Don't draw individual plots
res.nglFrame               = False     # Don't advance frame

#
# Note: the data in this case are already on a mercator projection.
#
# So, as long as you have the map projection set up correctly, you
# don't need to set sfXArray/sfYArray. You can instead set:
#
#   res.tfDoNDCOverlay = True
#
# telling PyNGL that the data doesn't need to go through transformation
# equations before being overlaid on a mercator projection.
#
res.sfYArray               = lat       # Where to overlay  
res.sfXArray               = lon       # contours on map

res.cnFillOn               = True          # Turn on contour fill
res.cnLinesOn              = False         # Turn off contour lines
res.cnFillDrawOrder        = "PostDraw"    # Draw contours last

res.cnLevelSelectionMode   = "ExplicitLevels"   # Set explicit contour levels
res.cnLevels               = numpy.arange(0,75,4)      # 0,5,10,...,70
res.cnLineLabelsOn         = False

res.nglSpreadColors        = True             # Use full color map
res.nglSpreadColorEnd      = -2               # Don't use added gray
res.lbLabelBarOn           = False            # Turn off labelbar.

res.mpProjection           = "Mercator"       # Projection
res.mpLimitMode            = "Corners"        # Method to zoom
res.mpLeftCornerLatF       = min(lat)
res.mpLeftCornerLonF       = min(lon)
res.mpRightCornerLatF      = max(lat)
res.mpRightCornerLonF      = max(lon)

res.mpFillOn               = True
res.mpFillColors           = [0,0,igray,igray]
res.mpGridLatSpacingF      = 5.
res.mpGridLineDashPattern  = 2

res.pmTickMarkDisplayMode  = "Never"

map_res = ["LowRes", "MediumRes", "HighRes"]   # Map coastline resolutions

#
# The HighRes map is only generated if PyNGL can find the RANGS database.
#  If the PYNGL_RANGS environmentl variable is set, then PyNGL will look
# in this directory. Otherwise, it will look in the default PyNGL
# directory. (The Ngl.pynglpath function is used to determine the
# directory.)
#
# Make sure the rangs database exists before we try to generate this frame.
#
rangs_dir = Ngl.pynglpath("rangs")           # Location of RANGS dir.

if(os.path.exists(rangs_dir)):
  nmap = 3
else:
  nmap = 2
  print "Sorry, you do not have the RANGS database installed."
  print "Will not generate the third frame of this example."

plot = []
for i in range(nmap):
  res.mpDataBaseVersion = map_res[i]
  res.tiMainString      = "Resolution = '" + map_res[i] + "'"
  plot.append(Ngl.contour_map (wks,fsd,res))  # Create plot, but don't draw

#
# Set some panel resources: a common labelbar and title.
# "[1,3]" indicates 1 row, 3 columns.
#
panelres = Ngl.Resources()
panelres.txString         = "Comparison of coastline resolutions"
panelres.nglPanelLabelBar = True   # Common labelbar
Ngl.panel(wks,plot,[1,3],panelres)

Ngl.end()
