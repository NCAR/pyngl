#
#  Import NumPy.
#
import Numeric

#
#  Import Ngl support functions.
#
from Ngl import *

#
#  Import all names from the NetCDF module.
#
from Scientific.IO.NetCDF import *

#
#  Open a netCDF file containing the Chesapeake Bay data.
#
dirc  = ncargpath("data")
cfile = NetCDFFile(dirc + "/cdf/ctcbay.nc","r")

#
#  Read the lat/lon/ele/depth arrays to Numeric arrays.
#
lat   = cfile.variables["lat"][:]
lon   = cfile.variables["lon"][:]
ele   = cfile.variables["ele"][:]
depth = cfile.variables["depth"][:]

#
#  Select a colormap and open an X11 window.
rlist = Resources()
rlist.wkColorMap = "rainbow+gray"
wks = ngl_open_wks("x11","chkbay",rlist)

#
#  The next set of resources will apply to the contour plot.
#
resources = Resources()

resources.nglSpreadColorStart = 15
resources.nglSpreadColorEnd   = -2 

resources.sfXArray         = lon  # Portion of map on which to overlay
resources.sfYArray         = lat  # contour plot.
resources.sfElementNodes   = ele
resources.sfFirstNodeIndex = 1

resources.cnFillOn         = True 
resources.cnLinesOn        = False
resources.cnLineLabelsOn   = False

resources.lbOrientation    = "Vertical"

contour = ngl_contour(wks,depth,resources)

#
#  The next set of resources will apply to the map plot.
#
resources.mpProjection          = "CylindricalEquidistant"
resources.mpDataBaseVersion     = "HighRes"     # "MediumRes" will run faster.

#
# Retrieve the actual lat/lon end points of the scalar array so
# we know where to overlay on map.
#
xs = ngl_get_float(contour.sffield,"sfXCActualStartF")
xe = ngl_get_float(contour.sffield,"sfXCActualEndF")
ys = ngl_get_float(contour.sffield,"sfYCActualStartF")
ye = ngl_get_float(contour.sffield,"sfYCActualEndF")

resources.mpLimitMode           = "LatLon"
resources.mpMinLonF             = xs     # -77.3244
resources.mpMaxLonF             = xe     # -75.5304
resources.mpMinLatF             = ys     #  36.6342
resources.mpMaxLatF             = ye     #  39.6212

resources.tiMainString          = "                    Chesapeake Bay~C~Bathymetry                               meters"
resources.pmTitleZone           = 3

resources.lbLabelFontHeightF    = 0.02
resources.lbAutoManage          = False

resources.pmTickMarkDisplayMode = "Always"

map = ngl_contour_map(wks,depth,resources)

ngl_end()
