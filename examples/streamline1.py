#
#  File:
#    streamline1.py
#
#  Synopsis:
#    Draws streamlines on a map over water only.
#
#  Category:
#    Streamlines on a map.
#
#  Author:
#    Mary Haley
#  
#  Date of original publication:
#    December, 2004
#
#  Description:
#    This example draws streamlines over water on a map using a
#    Cylindrical Equidistant map projection.  The "add_cyclic"
#    function is illustrated graphically.
#
#  Effects illustrated:
#    o  Streamlines over maps.
#    o  Adding cyclic points.
#    o  Specifying colors by name.
#    o  Polylines.
#    o  Masking land areas.
# 
#  Output:
#    This example produces two visualizations:
#      1.)  Streamlines on a Cylindrical Equidistant map over water
#           only highlighting missing cyclic points.
#      2.)  Same as 1.) with the cyclic points added.
#
#  Notes:
#     

#
#  Import Nio for reading netCDF files.
#
import Nio

#
#  Import Ngl support functions.
#
import Ngl

import os
#
#  Open the netCDF file.
#
file = Nio.open_file(os.path.join(Ngl.pynglpath("data"),"cdf","pop.nc"))

#
#  Open a workstation.
#
wks_type = "ps"

rlist            = Ngl.Resources()
rlist.wkColorMap = ["White","Black","Tan1","SkyBlue","Red"]
wks = Ngl.open_wks(wks_type,"streamline1",rlist)

#
#  Get the u/v and lat/lon variables.
#
urot  = file.variables["urot"]
vrot  = file.variables["vrot"]
lat2d = file.variables["lat2d"]
lon2d = file.variables["lon2d"]

#
# Set up resource list.
#
resources = Ngl.Resources()

#
# Don't advance frame, because we want to draw a couple of lines on 
# plot later.
#
resources.nglFrame = False

#
# Coordinate arrays for data
#
resources.vfXArray = lon2d[::4,::4]
resources.vfYArray = lat2d[::4,::4]

resources.mpProjection           = "CylindricalEquidistant"
resources.mpFillOn               = True
resources.mpLandFillColor        = "Tan1"
resources.mpOceanFillColor       = "SkyBlue"
resources.mpInlandWaterFillColor = "SkyBlue"
resources.mpGridAndLimbOn        = False 

resources.tiMainString           = "Streamline plot without cyclic point added"

plot = Ngl.streamline_map(wks,urot[::4,::4],vrot[::4,::4],resources) 

#
# Add a couple of lines showing the area where there's a gap in the
# data because of lack of a cyclic point.  (It should be obvious even
# without the lines.)
#
line_res                   = Ngl.Resources()      # line resources
line_res.gsLineColor       = "Red"                # line color
line_res.gsLineThicknessF  = 1.5                  # line thickness scale
line_res.gsLineDashPattern = 2                    # dashed lines

Ngl.polyline(wks,plot,lon2d[::4,0],lat2d[::4,0],line_res) 
Ngl.polyline(wks,plot,lon2d[::4,-1],lat2d[::4,-1],line_res) 

#
# Add a text string explaining the lines.
#
text_res                   = Ngl.Resources()      # text resources
text_res.txFontHeightF     = 0.03                 # font height
text_res.txFontColor       = "Red"

Ngl.text_ndc(wks,"dashed red line shows area with no data",0.5,0.17,text_res)

Ngl.frame(wks)                                    # Now advance frame.

#
# Add cyclic points.  Since lat2d/lon2d are 2D arrays, make them
# cyclic the same way you do the 2D data array.
#
u   = Ngl.add_cyclic(urot[::4,::4])
v   = Ngl.add_cyclic(vrot[::4,::4])
lon = Ngl.add_cyclic(lon2d[::4,::4])
lat = Ngl.add_cyclic(lat2d[::4,::4])

#
# Specify new coordinate arrays for data.
#
resources.vfXArray     = lon
resources.vfYArray     = lat

resources.tiMainString = "Streamline plot with cyclic point added"

plot = Ngl.streamline_map(wks,u,v,resources) 

#
# Add a couple of lines showing the area where the missing data were.
# Make the lines solid so we can see them.
#
line_res.gsLineDashPattern = 0

Ngl.polyline(wks,plot,lon2d[::4,0],lat2d[::4,0],line_res) 
Ngl.polyline(wks,plot,lon2d[::4,-1],lat2d[::4,-1],line_res) 

#
# Add a text string explaining the lines.
#
Ngl.text_ndc(wks,"red line shows area that previously had no data",0.5,0.17,text_res)

Ngl.frame(wks)

Ngl.end()
