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
wks_type = "ps"
wks = ngl_open_wks(wks_type,"chkbay",rlist)

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

#
# If you want high resolution map coastlines,download the RANGS/GSHHS
# files from:
#
#     http://www.io-warnemuende.de/homepages/rfeistel/index.html
#
# The files you need are:
#
#   rangs(0).zip    gshhs(0).zip
#   rangs(1).zip    gshhs(1).zip
#   rangs(2).zip    gshhs(2).zip
#   rangs(3).zip    gshhs(3).zip
#   rangs(4).zip    gshhs(4).zip
#
# Once you unzip these files, put them in the directory
# $python_prefx/pythonx.y/site-packages/PyNGL/ncarg/database/rangs
#
# Now you can change the following resource to "HighRes".
#
resources.mpDataBaseVersion     = "MediumRes"

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
