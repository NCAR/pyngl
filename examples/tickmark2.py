#
#  File:
#    tickmark2.py
#
#  Synopsis:
#    Illustrates how to change the tickmark labels on a map plot.
#
#  Categories:
#    tickmarks
#    maps
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    July 2011
#
#  Description:
#    This example also shows how to overlay a blank plot on a map.
#
#  Effects illustrated:
#    o  Customizing tickmark labels
#    o  Creating a blank plot
#    o  Overlaying plots
#
#  Output:
#     This example produces two frames: one with the default
#     map tickmarks and one with customized tickmarks.
#

# Import necessary modules
import numpy, os, Nio, Ngl

# Open data file
data_dir = Ngl.pynglpath("data")
cdffile  = Nio.open_file(os.path.join(data_dir,"cdf","941110_P.cdf"),"r")

# Read data
psl      = cdffile.variables["Psl"]   
psl_lon  = cdffile.variables["lon"][:]
psl_lat  = cdffile.variables["lat"][:]

# Open workstation
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"tickmark2")
Ngl.define_colormap(wks,"BlueDarkRed18")

# Set resources
res                   = Ngl.Resources()

#---Define X/Y values of data.
res.sfXArray          = psl_lon
res.sfYArray          = psl_lat

#---Contour resources
res.cnFillOn             = True
res.cnLineLabelsOn       = False

res.cnLevelSelectionMode = "ManualLevels"
res.cnMinLevelValF       = 950
res.cnMaxLevelValF       = 1040
res.cnLevelSpacingF      =    5

#---Map resources
res.mpGridAndLimbOn   = False

#---Labelbar resources
res.pmLabelBarOrthogonalPosF = -0.05
res.lbOrientation            = "Horizontal"

#---Main title
res.tiMainString      = "Default map tickmarks w/degree symbols"
res.tiMainFontHeightF = 0.02

# Create and draw contour-over-map plot
map = Ngl.contour_map(wks,psl,res)


# Turn off map tickmarks and recreate plot, but don't draw it.
res.nglDraw               = False
res.nglFrame              = False
res.pmTickMarkDisplayMode = "Never"
res.tiMainString          = "Customized map tickmarks w/o degree symbols"

map = Ngl.contour_map(wks,psl,res)

#----------------------------------------------------------------------
# This section creates the lat/lon labels we want for the customized
# map tickmarks.
#----------------------------------------------------------------------

#---Create arrays for where we want tickmarks on lat/lon axes.
lat_values = range(-90,100,30)
lon_values = range(-180,210,30)

#
# Create arrays of labels for the lat tickmark locations.
# Customize them with 'S' or 'N' as appropriate.
#
lat_labels = []
for l in lat_values:
  if l < 0:
    lat_labels.append("%gS" % abs(l))
  elif l > 0:
    lat_labels.append("%gN" % l)
  else:
    lat_labels.append("%g" % l)

#
# Create arrays of labels for the lon tickmark locations.
# Customize them with 'E' or 'W' as appropriate.
#
lon_labels  = []
for l in lon_values:
  if l < 0:
    lon_labels.append("%gW" % abs(l))
  elif l > 0:
    lon_labels.append("%gE" % l)
  else:
    lon_labels.append("%g" % l)

#----------------------------------------------------------------------
# This section creates a blank plot, for the purpose of customizing
# its tickmarks.
#----------------------------------------------------------------------

#---Resource list for blank plot
bres = Ngl.Resources()

#---Retrieve viewport coordinates and set them for blank plot.
bres.vpXF      = Ngl.get_float(map,"vpXF")
bres.vpYF      = Ngl.get_float(map,"vpYF")
bres.vpHeightF = Ngl.get_float(map,"vpHeightF")
bres.vpWidthF  = Ngl.get_float(map,"vpWidthF" )

#---Retrieve min/max values of map and set them for blank plot.
bres.trXMinF   = Ngl.get_float(map,"trXMinF")
bres.trXMaxF   = Ngl.get_float(map,"trXMaxF")
bres.trYMinF   = Ngl.get_float(map,"trYMinF")
bres.trYMaxF   = Ngl.get_float(map,"trYMaxF")

#---Default is inward.
bres.nglPointTickmarksOutward = True

#---Set the values and labels for the X axis of blank plot.
bres.tmXBMode                = "Explicit"
bres.tmXBValues              = lon_values
bres.tmXBLabels              = lon_labels
bres.tmXBLabelFontHeightF    = 0.015        # Make labels smaller. This
                                            # will affect Y labels too.

#---Set the values and labels for the Y axis of blank plot.
bres.tmYLMode                = "Explicit"
bres.tmYLValues              = lat_values
bres.tmYLLabels              = lat_labels

#---Align four corners of both plots, that is, don't do data transformation
bres.tfDoNDCOverlay          = True  

#---Create the blank plot.
blank = Ngl.blank_plot(wks,bres)

#---Draw blank plot for debugging purposes.
#Ngl.draw(blank)
#Ngl.frame(wks)

#---Overlay blank plot on existing map plot.
Ngl.overlay(map.base,blank)

#---Resize map so it fits within frame
Ngl.maximize_plot(wks, map)

#---Drawing the original map also draws the overlaid blank plot
Ngl.draw(map)
Ngl.frame(wks)


Ngl.end()
