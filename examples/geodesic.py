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
dirc  = Ngl.ncargpath("data")
cfile = NetCDFFile(dirc + "/cdf/hswm_d000000p000.g2.nc","r")

#
#  Read the grid centers and the kinetic energy into local variables.
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
rlist            = Ngl.Resources()
rlist.wkColorMap = "gui_default"
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"geodesic",rlist)

#
#  The next set of resources will apply to the contour plot and the labelbar.
#
resources = Ngl.Resources()

resources.sfXArray          = x
resources.sfYArray          = y
resources.sfXCellBounds     = cx
resources.sfYCellBounds     = cy

resources.cnFillOn          = True
resources.cnLinesOn         = False
resources.cnFillMode        = "RasterFill"
resources.cnLineLabelsOn    = False
resources.cnMaxLevelCount   = 22

resources.lbBoxLinesOn      = False
resources.lbTitleString     = "kinetic energy"
resources.tiMainString      = "2562 Element Geodesic grid"

resources.nglDraw           = False   # Just create the plot. Don't
resources.nglFrame          = False   # draw it or advance the frame.

contour = Ngl.contour(wks,ke,resources)

#
# Retrieve the actual lat/lon end points of the scalar array so
# we know where to overlay on map.
#
xs = Ngl.get_float(contour.sffield,"sfXCActualStartF")
xe = Ngl.get_float(contour.sffield,"sfXCActualEndF")
ys = Ngl.get_float(contour.sffield,"sfYCActualStartF")
ye = Ngl.get_float(contour.sffield,"sfYCActualEndF")

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

map = Ngl.contour_map(wks,ke,resources)

resources.cnRasterSmoothingOn = True
resources.tiMainString        = "Smooth raster contouring"

map = Ngl.contour_map(wks,ke,resources)

resources.cnFillMode   = "AreaFill"
resources.tiMainString = "Area fill contouring"

map = Ngl.contour_map(wks,ke,resources)

Ngl.end()
