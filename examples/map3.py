#
#  File:
#    map3.py
#
#  Synopsis:
#    Illustrates how to add your map tickmark labels and grid lines
#
#  Categories:
#    maps only
#    polylines
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    February, 2015
#
#  Description:
#    This example shows how to use a blank plot and polylines to add your own
#    map tickmark labels and grid lines.
#
#  Effects illustrated:
#    o  Creating a blank plot.
#    o  Adding annotations.
#    o  Polylines.
#    o  Using "getvalues" to retrieve resource values
# 
#  Output:
#     Two visualizations are produced.
#
#  Notes:
#     
import numpy as np
import Ngl

#----------------------------------------------------------------------
# Function for adding custom map tickmark labels to an existing
# PyNGL map object.
#
# This function turns off the actual tickmarks and only draws the
# labels, but you can opt to leave these on.
#----------------------------------------------------------------------
def add_map_tickmarks(wks,map,lat_spc,lon_spc):

  bres                   = Ngl.Resources()
  bres.nglMaximize       = False

#---Set some resources based on map plot already created.
  bres.vpXF              = Ngl.get_float(map, "vpXF")
  bres.vpYF              = Ngl.get_float(map, "vpYF")
  bres.vpWidthF          = Ngl.get_float(map, "vpWidthF")
  bres.vpHeightF         = Ngl.get_float(map, "vpHeightF")
  bres.trXMinF           = Ngl.get_float(map, "trXMinF")
  bres.trXMaxF           = Ngl.get_float(map, "trXMaxF")
  bres.trYMinF           = Ngl.get_float(map, "trYMinF")
  bres.trYMaxF           = Ngl.get_float(map, "trYMaxF")

  bres.tmEqualizeXYSizes = True   # make sure labels same size

#---Create longitude labels based on longitude spacing given.
  lon_values = np.arange(-180,181,lon_spc)
  lon_labels = []
  for l in lon_values:
    if l < 0:
      lon_labels.append("%i~S~o~N~W" % np.fabs(l))
    elif l > 0:
      lon_labels.append("%i~S~o~N~E" % l)
    else:
      lon_labels.append("0")
  
#---Create latitude labels based on latitude spacing given.
  lat_values = np.arange(-90,91,lat_spc)
  lat_labels = []
  for l in lat_values:
    if l < 0:
      lat_labels.append("%i~S~o~N~S" % np.fabs(l))
    elif l > 0:
      lat_labels.append("%i~S~o~N~N" % l)
    else:
      lat_labels.append("EQ" % l)

#---Set tickmark resources
  bres.tmXBMode                = "Explicit"
  bres.tmXBValues              = lon_values
  bres.tmXBLabels              = lon_labels
  
  bres.tmYLMode                = "Explicit"
  bres.tmYLValues              = lat_values
  bres.tmYLLabels              = lat_labels
  
  bres.tmYLLabelFontHeightF    = 0.009    # Make these labels smaller.
  bres.tmXBLabelFontHeightF    = 0.009    # Ditto
  
#---To urn on tickmark lines, change these values to something like 0.01
#---Turn off tickmark lines
  bres.tmXBMajorLengthF        = 0.
  bres.tmYLMajorLengthF        = 0.
  bres.tmXBMajorOutwardLengthF = 0.
  bres.tmYLMajorOutwardLengthF = 0.
  
#---Create the blank plot with the special labels
  blank  = Ngl.blank_plot(wks,bres)

#---Attach blank plot with special labels to map plot, and return
  sres                = Ngl.Resources()
  sres.amZone         = 0     # '0' means centered over base plot.
  sres.amResizeNotify = True
  return Ngl.add_annotation(map,blank,sres)

#----------------------------------------------------------------------
# Function for adding custom lat/lon grid lines to an existing
# PyNGL map object.
#
# You only need to use this function if mpGridLatSpacingF and/or
# mpGridLonSpacingF are not working as expected.
#----------------------------------------------------------------------
def add_map_gridlines(wks,map,lat_spc,lon_spc):

#---Set some resources for the polyline
  lnres                   = Ngl.Resources()
  lnres.gsLineColor       = "Gray25"
  lnres.gsLineThicknessF  = 1.0     # 1.0  is the default
  lnres.gsLineDashPattern = 2       # 0 (solid) is the default

  lon_values = np.arange(-180,181,lon_spc)
  lat_values = np.arange(-90,91,lat_spc)

#---Add the line to existing plot
  lines = []
  for lat in lat_values:
    lines.append(Ngl.add_polyline(wks,map,[-180,0,180],[lat,lat,lat],lnres))
  for lon in lon_values:
    lines.append(Ngl.add_polyline(wks,map,[lon,lon],[-90,90],lnres))
  lnres.gsLineColor = "red"

  return lines

#----------------------------------------------------------------------
# Main code
#----------------------------------------------------------------------

lon_spacing = 40
lat_spacing = 20

wks_type = "png"
wks = Ngl.open_wks(wks_type,"map3")

#---Create and draw map; note how lat/lon spacing doesn't line up with tickmarks
res                       = Ngl.Resources()
res.mpGridLatSpacingF     = lat_spacing
res.mpGridLonSpacingF     = lat_spacing
res.tiMainString          = "Default map tickmarks and grid lines"

map = Ngl.map(wks,res)
del res

#---Create map again, but turn off grid lines and tickmarks
res                       = Ngl.Resources()

res.nglDraw               = False     # Don't draw map
res.nglFrame              = False     # Don't advance frame
res.mpGridAndLimbOn       = False     # Will add own grid lines
res.pmTickMarkDisplayMode = "Never"   # Will add own labels
res.tiMainString          = "Custom map labels and grid lines"

map = Ngl.map(wks,res)     # Won't get drawn just yet

#---Add custom tickmarks and grid lines
labels = add_map_tickmarks(wks,map,lat_spacing,lon_spacing)
lines  = add_map_gridlines(wks,map,lat_spacing,lon_spacing)

#---Resize plot and draw
Ngl.maximize_plot(wks,map)

Ngl.draw(map)
Ngl.frame(wks)

Ngl.end()
