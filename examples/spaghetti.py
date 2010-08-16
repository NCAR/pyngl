#
#  File:
#    spahgetti.py
#
#  Synopsis:
#    Illustrates generating one contour per time step to
#    make a spahgetti-style plot.
#
#  Category:
#    Contouring over maps.
#
#  Author:
#    Mary Haley (taken from a CCSM NCL example)
#  
#  Date of initial publication:
#    December, 2006
#
#  Description:
#    This example illustrates generating one contour level per
#    time step, and overlaying each contour on a polar map.
#
#  Effects illustrated:
#    o  Selecting a color map by name.
#    o  Adding a color to an existing color map.
#    o  Reading from a NetCDF file.
#    o  Adding a cyclic point in the longitude dimension.
#    o  Creating a polar map projection.
#    o  Overlaying multiple plots on a single plot.
#    o  Adding additional text to a plot.
#    o  Copying one resource list to another.
# 
#  Output:
#    This example produces a single frame.
#
#  Notes:
#     

import numpy
import Ngl
import Nio
import os

#
# Open file and get variable. The lat/lon variables will be
# generated using fspan.  This data came from another dataset
# that had lat/lon on the file, but lat/lon was nothing more
# than equally-spaced values which we can regenerate exactly.
#
f = Nio.open_file(os.path.join(Ngl.pynglpath("data"),"cdf","hgt.nc"),"r")

hgt = f.variables["HGT"][:,:,:]
lat = Ngl.fspan(-90,90,73)
lon = Ngl.fspan(0,357.5,144)

# Add a cyclic point in the longitude dimension.
hgt0,lon = Ngl.add_cyclic(hgt[0,:,:],lon)

#
# Start graphics.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"spaghetti")
Ngl.define_colormap(wks,"default")         # Change color map.

red   = 245./255.
green = 222./255.
blue  = 179./255.
tan = Ngl.new_color(wks,red,green,blue)   # Add tan.

mpres = Ngl.Resources()

mpres.nglDraw              = False       # Do not draw until the end.
mpres.nglFrame             = False       # Do not automatically advance frame.

#
# Set contour resources first, because we'll make a copy of them
# to use later for when we generate contours only.
#
mpres.sfXArray             = lon         # Area of map to overlay 
mpres.sfYArray             = lat         # contours on.

mpres.cnLevelSelectionMode = "ExplicitLevels" # explicit contour levels
mpres.cnLevels             = 5500             # which level(s) to plot
mpres.cnInfoLabelOn        = False            # no info label
mpres.cnLineLabelsOn       = False            # no line labels

mpres.cnLineColor          = 1                # use numerical indices
mpres.cnLineThicknessF     = 2                # thickness of contour lines

mpres.tiMainString         = "Spaghetti-style contours"    # title 
mpres.tiMainFontHeightF    = 0.02

mpres.mpProjection          = "Stereographic"
mpres.mpEllipticalBoundary  = True

mpres.mpLimitMode           = "LatLon"   # Specify area of map
mpres.mpMaxLatF             = 90.        # to zoom in on.
mpres.mpMinLatF             = 30
mpres.mpCenterLatF          = 90.

mpres.mpFillOn              = True
mpres.mpFillColors          = [0,-1,tan,-1]     # -1 is transparent
mpres.mpOutlineOn           = False
mpres.mpGridLineDashPattern = 2

mpres.pmTickMarkDisplayMode = "Never"    # Turn off default ticmkark object.

#
# Create base plot upon which we will overlay the rest of the
# "spaghetti" contours.
#
plot_base = Ngl.contour_map(wks,hgt0,mpres)

# 
# Copy just the contour resources from mpres to a new resource list (cnres).
#
cnres = Ngl.Resources()
for t in dir(mpres):
  if (t[0:2] == 'cn' or t[0:2] == 'sf' or t[0:3] == 'ngl'):
    setattr(cnres,t,getattr(mpres,t))

#
# Loop over other 19 fields but only do contour plot.
# Note the color index is changing. In loop, we select
# a new color from the default color map.
#
for i in range(19):
  cnres.cnLineColor = 2+i                      # Change line color.
  hgt0 = Ngl.add_cyclic(hgt[i+1,:,:])          # Add a cyclic pt to data.
  plot = Ngl.contour(wks,hgt0,cnres)           # Generate contours.
  Ngl.overlay(plot_base,plot)                  # Overlay this contour on map.

Ngl.draw(plot_base)                            # Draw map.

# Add some subtitles at top.

txres               = Ngl.Resources()
txres.txFontHeightF = 0.02

txres.txJust = "CenterLeft"
Ngl.text_ndc(wks,"Geopotential Height",0.01,0.9,txres)

txres.txJust = "CenterRight"
Ngl.text_ndc(wks,"gpm",0.95,0.9,txres)

Ngl.frame(wks)        # Now advance the frame.

Ngl.end()

