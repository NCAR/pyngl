#
#  Import NumPy.
#
import Numeric

#
#  Import Ngl support functions.
#
import Ngl

#
#  Import all names from the NetCDF module.
#
from Scientific.IO.NetCDF import NetCDFFile

#
#  Open a netCDF file containing the Chesapeake Bay data.
#
#  This data is from the Chesapeake Community Model Program Quoddy
#  model:
#
#    http://ccmp.chesapeake.org
#
#  using the NOAA/NOS standardized hydrodynamic model NetCDF format:
#
#    https://sourceforge.net/projects/oceanmodelfiles
#
dirc  = Ngl.ncargpath("data")
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
#
rlist            = Ngl.Resources()
rlist.wkColorMap = "rainbow+gray"
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"chkbay",rlist)

#
#  The next set of resources will apply to the contour plot.
#
resources = Ngl.Resources()

resources.nglSpreadColorStart = 15
resources.nglSpreadColorEnd   = -2 

resources.sfXArray         = lon  # Portion of map on which to overlay
resources.sfYArray         = lat  # contour plot.
resources.sfElementNodes   = ele
resources.sfFirstNodeIndex = 1

resources.cnFillOn         = True 
resources.cnLinesOn        = False
resources.cnLineLabelsOn   = False

#
# This plot isn't very interesting because it isn't overlaid on a map.
# We are only creating it so we can retrieve information that we need
# to overlay it on a map plot later. You can turn off this plot
# by setting the nglDraw and nglFrame resources to False.
#
contour = Ngl.contour(wks,depth,resources)

#
#  The next set of resources will apply to the map plot.
#
resources.mpProjection = "CylindricalEquidistant"

#
# If you want high resolution map coastlines, download the RANGS/GSHHS
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
resources.mpDataBaseVersion = "MediumRes"

#
# Retrieve the actual lat/lon end points of the scalar array so
# we know where to overlay on map.
#
xs = Ngl.get_float(contour.sffield,"sfXCActualStartF")
xe = Ngl.get_float(contour.sffield,"sfXCActualEndF")
ys = Ngl.get_float(contour.sffield,"sfYCActualStartF")
ye = Ngl.get_float(contour.sffield,"sfYCActualEndF")

resources.mpLimitMode           = "LatLon"
resources.mpMinLonF             = xs     # -77.3244
resources.mpMaxLonF             = xe     # -75.5304
resources.mpMinLatF             = ys     #  36.6342
resources.mpMaxLatF             = ye     #  39.6212

#
# In the chkbay.res file, a resource is being set to indicate the "~"
# character is to represent a function code. A function code signals an
# operation you want to apply to the following text.  In this case,
# ~H10Q~ inserts 10 horizontal spaces before the text, and ~C~ causes
# a line feed (carriage return.
#

resources.tiMainString       = "~H10Q~Chesapeake Bay~C~Bathymetry~H16Q~meters"
resources.lbLabelFontHeightF = 0.02

map = Ngl.contour_map(wks,depth,resources)

Ngl.end()
