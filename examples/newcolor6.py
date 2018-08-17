#
#  File:
#    newcolor6.py
#
#  Synopsis:
#    Illustrates applying opacity to filled contours to emphasize a
#    particular area
#
#  Categories:
#    contour plots
#
#  Author:
#    Mary Haley, based on NCL example
#  
#  Date of initial publication:
#    September 2017
#
#  Description:
#    This example shows how to emphasize an area in a contour plot
#    using opacity (transparency).
#
#  Effects illustrated:
#    o  Showing features of the new color display model
#    o  Drawing partially transparent filled contours
#    o  Using opacity to emphasize or subdue overlain features
#    o  Using cnFillPalette to assign a color palette to contours
#    o  Using polygon fill in contour plot
# 
#  Output:
#     Two visualizations
#

import numpy,os
import Ngl,Nio

# Set resources common for contour plots
def set_common_resources():
  res                        = Ngl.Resources()

  res.nglFrame               = False
  res.nglDraw                = False

  res.cnLevelSelectionMode   = "ExplicitLevels"
  res.cnLevels               = numpy.arange(-12,42,2)

  res.cnFillOn               = True
  res.cnFillPalette          = "BlueYellowRed"

  res.cnLinesOn              = False
  res.cnLineLabelsOn         = False
  res.cnInfoLabelOn          = False

  res.lbOrientation          = "Horizontal"

  return(res)

# Read in zonal winds
f    = Nio.open_file("$NCARG_ROOT/lib/ncarg/data/cdf/uv300.nc","r")
u    = f.variables["U"]
lat  = f.variables["lat"]
lon  = f.variables["lon"]


# Start the graphics
wks_type = "png"
wks = Ngl.open_wks (wks_type,"newcolor6")

res = set_common_resources()

# Set resources for contour/map plot
bres = set_common_resources()

bres.mpFillOn          = False 
bres.tiMainString      = "Use transparency to emphasize a particular area"
bres.tiMainFontHeightF = 0.018
bres.cnFillOpacityF    = 0.5     # Half transparent

unew,lonnew = Ngl.add_cyclic(u[1,:,:],lon[:])

bres.sfMissingValueV   = u._FillValue
bres.sfXArray          = lonnew[:]
bres.sfYArray          = lat[:]

base_plot = Ngl.contour_map(wks,unew,bres)

# Set resources for contour plot only
ores = set_common_resources()

ores.cnFillOpacityF        = 1.0     # Fully opaque
ores.lbLabelBarOn          = False   # Turn off labelbar
ores.sfMissingValueV       = u._FillValue

ores.sfXArray        = lon['lon|-120:120']
ores.sfYArray        = lat['lat|-30:30']

# Use PyNIO's selection syntax to subset the lat/lon area of interest
overlay_plot = Ngl.contour(wks,u['time|i1 lat|-30:30 lon|-120:120'],ores)

# Overlay the contour plot on the contour/map plot
Ngl.overlay(base_plot,overlay_plot)

# Drawing the base plot draws both plots
Ngl.draw(base_plot)
Ngl.frame(wks)

# Create the contour/plot again, but with no transparency
del bres.cnFillOpacityF
plot = Ngl.contour_map(wks,u[1,:,:],bres)

# Set resources for a partially transparent polygon.
gnres                = Ngl.Resources()
gnres.gsFillOpacityF = 0.6          # mostly opaque
gnres.gsFillColor    = "white"

lat_box = [ -30,-30, 30,  30, -30]
lon_box = [-120,120,120,-120,-120]

# Add a partially opaque filled polygon box to the filled contours
gsid = Ngl.add_polygon(wks,plot,lon_box,lat_box,gnres)

# This draws the filled contours with the partially opaque box on top.
Ngl.draw(plot)
Ngl.frame(wks)
 
Ngl.end()

