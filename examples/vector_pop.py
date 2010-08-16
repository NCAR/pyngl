#
#  File:
#    vector_pop.py
#
#  Synopsis:
#    Shows how to plot vectors on a POP grid with land masked.
#
#  Category:
#    Vectors over maps
#
#  Author:
#    Dave Brown
#  
#  Date of initial publication:
#    October, 2005
#
#  Description:
#    This example shows how to draw vectors on a POP grid, and 
#    shows the various ways you can set vector resources to thin 
#    the vectors, change their lengths, and turn them into curly 
#    vectors.  It also shows how to plot the underlying POP grid
#    structure using cell fill mode will a transparent fill color.
#
#  Effects illustrated:
#    o  Drawing vectors over maps using an orthographic projection.
#    o  Drawing curly vectors.
#    o  Using cell fill mode.
#    o  Using named colors.
#    o  Masking land.
#    o  Defining cyclic points.
# 
#  Output:
#    This example produces eight visualizations:
#      1.)  Plots the underlying POP grid structure using cell 
#           fill mode will a transparent fill color.
#      2.)  Draws a full set of vectors over water.
#      3.)  Same as 2.) but thins the vectors by drawing only
#           every third one.
#      4.)  Thins the vectors by using the vcMinDistanceF resource.
#      5.)  Same as 4.) but makes the vectors longer.
#      6.)  Same as 5.) but uses curly vectors.
#      7.)  Same as 6.) but uses a different color map and colors
#           the vectors by magnitude.
#      8.)  Same as 7.) but uses filled arrows.
#
#  Notes:
#     

#
#  Import Ngl support functions.
#
import Ngl
import Nio
import numpy
import os

#
#  Open the netCDF file.
#
dirc = Ngl.pynglpath("data")
file = Nio.open_file(os.path.join(dirc,"cdf","pop.nc"))

#
#  Open a workstation.
#
rlist            = Ngl.Resources()
rlist.wkColorMap = ["White","Black","Tan1","SkyBlue","Red"]
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"vector_pop",rlist)

#
#  Get the u/v and lat/lon variables.
#
urot  = file.variables["urot"]
vrot  = file.variables["vrot"]
lat2d = file.variables["lat2d"]
lon2d = file.variables["lon2d"]
t     = file.variables["t"]

u    = Ngl.add_cyclic(urot[290:])
v    = Ngl.add_cyclic(vrot[290:])
lon  = Ngl.add_cyclic(lon2d[290:])
lat  = Ngl.add_cyclic(lat2d[290:])
temp = Ngl.add_cyclic(t[290:])

#
# Set up resource list.
#
# First we're going to draw a contour plot with the grid cells outlined
# so you can see what the POP grid looks like.
#
cnres = Ngl.Resources()

cnres.nglFrame               = False
#
# Set coordinate arrays for data. This is necessary in order to overlay
# the contours on a map.
#
cnres.sfXArray               = lon[::3,::3]
cnres.sfYArray               = lat[::3,::3]

cnres.mpFillOn               = False
cnres.mpLandFillColor        = "Tan1"
cnres.mpOceanFillColor       = "SkyBlue"
cnres.mpInlandWaterFillColor = "SkyBlue"
cnres.mpLimitMode            = "LatLon"
cnres.mpCenterLonF           = -80.
cnres.mpCenterLatF           = 55
cnres.mpMinLatF              = 60
cnres.mpDataBaseVersion      = "MediumRes"

cnres.cnFillOn                      = True
cnres.cnLinesOn                     = False
cnres.cnLineLabelsOn                = False
cnres.cnFillMode                    = "CellFill"
cnres.cnCellFillEdgeColor           = "Black"
cnres.cnMonoFillColor               = True
cnres.cnFillColor                   = "Transparent"
cnres.cnCellFillMissingValEdgeColor = "Red"

cnres.lbLabelBarOn                  = False

cnres.tiMainString           = "Pop Grid cells -- grid stride [::3,::3]"

plot = Ngl.contour_map(wks,temp[:-2:3,:-2:3],cnres) 

Ngl.frame(wks)

#
# Now set up a vector resource variable.
#
vcres = Ngl.Resources()

#
# Set coordinate arrays for data. This is necessary in order to overlay
# the vectors on a map.
#
vcres.vfXArray = lon
vcres.vfYArray = lat

vcres.mpProjection           = "Orthographic"
vcres.mpFillOn               = True
vcres.mpLandFillColor        = "Tan1"
vcres.mpOceanFillColor       = "SkyBlue"
vcres.mpInlandWaterFillColor = "SkyBlue"
vcres.mpLimitMode            = "LatLon"
vcres.mpCenterLonF           = -80.
vcres.mpCenterLatF           = 55
vcres.mpMinLatF              = 60
vcres.mpDataBaseVersion      = "MediumRes"

vcres.vcMinDistanceF          = 0.0
vcres.vcRefAnnoString2        = urot.units
vcres.vcRefAnnoOrthogonalPosF = -0.18       # Move ref anno up into plot.

vcres.tiMainString            = "Currents at depth " + str(urot.z_t[0])

#
# Draw the full set of vectors.
#
plot = Ngl.vector_map(wks,u,v,vcres) 

#
# Now we're going to thin the vectors by drawing only every third one.
#
vcres.vfXArray           = lon[::3,::3]
vcres.vfYArray           = lat[::3,::3]
vcres.vcRefLengthF       = 0.1

vcres.tiMainString       = "Striding the grid [::3,::3]"

plot = Ngl.vector_map(wks,u[::3,::3],v[::3,::3],vcres) 

#
# This time thin the vectors by setting the vcMinDistanceF resource.
#
vcres.vfXArray           = lon
vcres.vfYArray           = lat

vcres.vcRefMagnitudeF    = 30.0
vcres.vcMinDistanceF     = 0.015
vcres.vcRefLengthF       = 0.08

vcres.tiMainString       = "vcMinDistanceF = 0.015"
plot = Ngl.vector_map(wks,u,v,vcres) 

#
# Make the vectors a little longer.
#
vcres.vcMinFracLengthF   = 0.1

vcres.tiMainString       = "vcMinFracLenF = 0.1"

plot = Ngl.vector_map(wks,u,v,vcres)

#
# Change over to curly vectors.
#
vcres.vcMinDistanceF     = 0.013
vcres.vcGlyphStyle       = "CurlyVector"

vcres.tiMainString = "vcGlyphStyle = 'CurlyVector'"

plot = Ngl.vector_map(wks,u,v,vcres) 

# 
# Change the color map and draw vectors colored by magnitude.
#
wkres = Ngl.Resources()
wkres.wkColorMap = "rainbow+gray"
Ngl.set_values(wks,wkres)

vcres.nglSpreadColorStart  = 24
vcres.nglSpreadColorEnd    = -2 # Don't include last color, which is gray

vcres.mpOceanFillColor     = "Transparent"
vcres.mpLandFillColor      = "Gray"

vcres.vcRefAnnoArrowLineColor = "Black"
vcres.vcMaxLevelCount      = 23
vcres.vcMonoLineArrowColor = False

vcres.tiMainString         = "Curly vectors colored by magnitude"

plot = Ngl.vector_map(wks,u,v,vcres) 

#
# Change to filled arrows.
#
vcres.vcGlyphStyle             = "FillArrow"
vcres.vcMonoFillArrowFillColor = False
vcres.vcRefMagnitudeF          = 0.0
vcres.vcRefAnnoOrthogonalPosF  = -0.20

vcres.tiMainString       = "Filled arrows"

plot = Ngl.vector_map(wks,u,v,vcres) 

Ngl.end()
