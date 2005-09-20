#
#  Import Nio for reading netCDF files.
#
import Nio

#
#  Import Ngl support functions.
#
import Ngl

#
#  Open the netCDF file.
#
file = Nio.open_file(Ngl.ncargpath("data") + "/cdf/pop.nc","r")

#
#  Open a workstation.
#
wks_type = "ps"

rlist            = Ngl.Resources()
rlist.wkColorMap = ["White","Black","Tan1","SkyBlue","Red"]
wks = Ngl.open_wks(wks_type,"streamline",rlist)

#
#  Get the u/v and lat/lon variables.
#
urot  = file.variables["urot"]
vrot  = file.variables["vrot"]
lat2d = file.variables["lat2d"][:,:]
lon2d = file.variables["lon2d"][:,:]

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
resources.vfXArray = lon2d
resources.vfYArray = lat2d

if hasattr(urot,"_FillValue"):
  resources.vfMissingUValueV = urot._FillValue
if hasattr(vrot,"_FillValue"):
  resources.vfMissingVValueV = vrot._FillValue

resources.mpProjection           = "CylindricalEquidistant"
resources.mpFillOn               = True
resources.mpLandFillColor        = "Tan1"
resources.mpOceanFillColor       = "SkyBlue"
resources.mpInlandWaterFillColor = "SkyBlue"
resources.mpGridAndLimbOn        = False 

resources.tiMainString           = "Streamline plot without cyclic point added"

plot = Ngl.streamline_map(wks,urot,vrot,resources) 

#
# Add a couple of lines showing the area where there's a gap in the
# data because of lack of a cyclic point.  (It should be obvious even
# without the lines.)
#
line_res                   = Ngl.Resources()      # line resources
line_res.gsLineColor       = "Red"                # line color
line_res.gsLineThicknessF  = 1.5                  # line thickness scale
line_res.gsLineDashPattern = 2                    # dashed lines

Ngl.polyline(wks,plot,lon2d[:,0],lat2d[:,0],line_res) 
Ngl.polyline(wks,plot,lon2d[:,-1],lat2d[:,-1],line_res) 

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
u   = Ngl.add_cyclic(urot)
v   = Ngl.add_cyclic(vrot)
lon = Ngl.add_cyclic(lon2d)
lat = Ngl.add_cyclic(lat2d)

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

Ngl.polyline(wks,plot,lon2d[:,0],lat2d[:,0],line_res) 
Ngl.polyline(wks,plot,lon2d[:,-1],lat2d[:,-1],line_res) 

#
# Add a text string explaining the lines.
#
Ngl.text_ndc(wks,"red line shows area that previously had no data",0.5,0.17,text_res)

Ngl.frame(wks)

Ngl.end()
