#
#  File:
#    geodesic.py
#
#  Synopsis:
#    Draws colored contours on a map using a geodesic grid.
#
#  Category:
#    Contours on maps
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    September, 2004
#
#  Description:
#    Reads in data defined on a geodesic grid (a grid of contiguous
#    hexagons) and uses raster fill to draw contours of kinetic
#    energy mapped onto an orthographic map projection.
#
#  Effects illustrated:
#    o How to define a grid by specifying grid centers and boundaries
#      of grid cells surrounding each grid center (in this case the
#      grid cells are hexagons).
#    o How to use raster fill mode for contouring.
#    o How to use area fill mode for contouring.
# 
#  Output:
#    Three visualizations are produced:
#      1.) Raster fill contours
#      2.) Smoothed raster fill contours
#      3.) Area fill contours
#
#  Notes:
#    1.)  For grid cells that are not triangles an algorithm is
#         employed for converting the cells to triangles before
#         contouring - direct contouring is available only for
#         triangular meshes at this time.
#
#    2.)  The grid for this example was provided by Dave Randall, 
#         Todd Ringler, and Ross Heikes of Colorado State University.
#         The data was downloaded from:
#
#           http://kiwi.atmos.colostate.edu/BUGS/geodesic/interpolate.html
#     

#
#  Import numpy.
#
import numpy

#
#  Import Ngl support functions.
#
import Ngl

#
#  Import Nio for reading netCDF files.
#
import Nio

#
# Open a netCDF file containing the geodesic grid and data on that grid.
#
# This grid came to us via Dave Randall, Todd Ringler, and Ross Heikes of
# CSU. The data for this mesh were originally downloaded from:
#
#   http://kiwi.atmos.colostate.edu:16080/BUGS/projects/geodesic/
#
# The above URL doesn't seem to be active anymore. Here's a new URL:
#
#   http://kiwi.atmos.colostate.edu/BUGS/geodesic/interpolate.html
#
dirc  = Ngl.pynglpath("data")
cfile = Nio.open_file(dirc + "/cdf/hswm_d000000p000.g2.nc","r")

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

resources.sfXArray          = x      # These four resources define
resources.sfYArray          = y      # the type of grid we want to 
resources.sfXCellBounds     = cx     # contour.
resources.sfYCellBounds     = cy

resources.cnFillOn          = True
resources.cnFillMode        = "RasterFill"
resources.cnLinesOn         = False
resources.cnLineLabelsOn    = False
resources.cnMaxLevelCount   = 22

resources.tiMainString      = "2562 Element Geodesic grid"

resources.lbBoxLinesOn      = False
resources.lbTitleString     = "kinetic energy"

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

resources.nglDraw           = True      # Now we want to draw the plot
resources.nglFrame          = True      # and advance the frame.

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
