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
cfile = NetCDFFile("../../users/dbrown/hswm_d000000p000.g2.nc","r")

#
#  Read the lat/lon/ele/depth arrays to Numeric arrays.
#
r2d = 57.2957795             # radians to degrees
x   = cfile.variables["grid_center_lon"][:] * r2d
y   = cfile.variables["grid_center_lat"][:] * r2d
cx  = cfile.variables["grid_corner_lon"][:] * r2d
cy  = cfile.variables["grid_corner_lat"][:] * r2d
ke  = cfile.variables["kinetic_energy"][2,:]

#
#  Select a colormap and open a workstation.
#
rlist            = Resources()
rlist.wkColorMap = "gui_default"
wks_type = "ps"
wks = ngl_open_wks(wks_type,"geodesic",rlist)

#
#  The next set of resources will apply to the contour plot and the labelbar.
#
resources = Resources()

resources.sfXArray          = x
resources.sfYArray          = y
resources.sfXCellBounds     = cx
resources.sfYCellBounds     = cy

resources.cnFillOn          = True
resources.cnLinesOn         = False
resources.cnFillMode        = "RasterFill"
resources.cnLineLabelsOn    = False
resources.cnMaxLevelCount   = 22

resources.lbLabelAutoStride = True
resources.lbBoxLinesOn      = False
resources.lbTitleString     = "kinetic energy"
resources.tiMainString      = "2562 Element Geodesic grid"

resources.nglDraw           = False   # Just create the plot. Don't
resources.nglFrame          = False   # draw it or advance the frame.

contour = ngl_contour(wks,ke,resources)

#
# Retrieve the actual lat/lon end points of the scalar array so
# we know where to overlay on map.
#
xs = ngl_get_float(contour.sffield,"sfXCActualStartF")
xe = ngl_get_float(contour.sffield,"sfXCActualEndF")
ys = ngl_get_float(contour.sffield,"sfYCActualStartF")
ye = ngl_get_float(contour.sffield,"sfYCActualEndF")

resources.nglDraw           = True
resources.nglFrame          = True
resources.mpGridAndLimbOn   = False
resources.mpProjection      = "Orthographic"
resources.mpDataBaseVersion = "MediumRes"
resources.mpLimitMode       = "LatLon"
resources.mpMinLonF         = xs
resources.mpMaxLonF         = xe
resources.mpMinLatF         = ys
resources.mpMaxLatF         = ye
resources.mpCenterLatF      =  40
resources.mpCenterLonF      = -100

map = ngl_contour_map(wks,ke,resources)

resources.cnRasterSmoothingOn = True
resources.tiMainString        = "Smooth raster contouring"

map = ngl_contour_map(wks,ke,resources)

resources.cnFillMode   = "AreaFill"
resources.tiMainString = "Area fill contouring"

map = ngl_contour_map(wks,ke,resources)

ngl_end()
