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
#    o  Reading from a NetCDF file.
#    o  Adding a cyclic point in the longitude dimension.
#    o  Creating a polar map projection.
#    o  Overlaying multiple plots on a single plot.
#    o  Adding additional text to a plot.
#    o  Copying one resource list to another.
#    o  Using indexed color.
# 
#  Output:
#    This example produces a single frame.
#
#  Notes:
#     

from __future__ import print_function
import numpy
import Ngl
import Nio
import os, sys

#----------------------------------------------------------------------
# This procedure adds longitude labels to the outside of a circular
# polar stereographic map.
#----------------------------------------------------------------------
def add_lon_labels(wks):
#
# List the longitude values where you want labels.  It's assumed that longitude=0
# is at the bottom of the plot, and 180W at the top. You can adjust as necessary.
#
  lon_values = numpy.arange(-180,180,30)

  nlon       = lon_values.shape[0]
  lat_values = numpy.zeros(nlon,'f') + mpres.mpMinLatF + 0.005

#
# Get the NDC coordinates of these lat,lon labels.
# We'll use this information to place labels *outside* of
# the map plot.
#
  xndc, yndc = Ngl.datatondc(plot_base,lon_values,lat_values)

#
# Set an array of justification strings to use with the "txJust" resource
# for each label, based on which quadrant it appears in.
#
  just_strs  = ["BottomCenter",                 # top of plot
                "BottomRight","BottomRight",    # upper left quadrant
                "CenterRight",                  # left of plot
                "TopRight","TopRight",          # lower left quadrant
                "TopCenter",                    # bottom of plot
                "TopLeft","TopLeft",            # lower right quadrant
                "CenterLeft",                   # right of plot
                "BottomLeft","BottomLeft"]      # upper right qudrant

# Create an array of longitude labels with "W" and "E" added.
  lon_labels = []
  for i in range(nlon):
    if lon_values[i] < 0:
      lon_labels.append("%gW" % abs(lon_values[i]))
    elif lon_values[i] > 0:
      lon_labels.append("%gE" % lon_values[i])
    else:
      lon_labels.append("%g" % lon_values[i])

# Loop through each label and add it.
  txres = Ngl.Resources()
  txres.txFontHeightF = 0.01
  for i in range(nlon):
    txres.txJust = just_strs[i]
    Ngl.text_ndc(wks,lon_labels[i],xndc[i],yndc[i],txres)

  return

#----------------------------------------------------------------------
# Main code
#----------------------------------------------------------------------

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
wks_type = "png"
wks = Ngl.open_wks(wks_type,"spaghetti")
Ngl.define_colormap(wks,"default")         # Change color map.

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
mpres.mpFillColors          = ["white","transparent","tan","transparent"]
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

# Draw the plot, add the lon labels, and advance the frame.
Ngl.draw(plot_base)
add_lon_labels(wks)
Ngl.frame(wks)

Ngl.end()

