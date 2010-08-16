#
#  File:
#    ctnccl.py
#
#  Synopsis:
#    Draws contours on a triangular mesh.
#
#  Category:
#    Contours on non-rectangular grids
#    Contours over maps
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    October, 2004
#
#  Description:
#    This example reads data defined on a triangular
#    mesh and creates a colored contour visualization of 
#    ocean depth off the coast of North Carolina.
#
#  Effects illustrated:
#    o  Reading from a NetCDF file using Nio.
#    o  How to spread colors evenly over a
#         subset of a color table.
#    o  How to specify explicit contour levels.
#    o  Using a cylindrical equidistant map projection.
#    o  How to specify map database resolution.
#    o  How to turn off tickmarks.
# 
#  Output:
#    A single visualization is produced.
#
#  Notes:
#    1.)  The data is from the Naval Research Laboratory at the Stennis
#         Space Center. For more information about this grid, see the
#         article "Application of a Shelf-Scale Model to Wave-Induced
#         Circulation: Rip Currents" by Mark Cobb and Cheryl Ann Blain,
#         of the Naval Research Lab, Stennis Space Center, MS, Ocean 
#         Dynamics and Prediction Branch:
#
#           http://www.stormingmedia.us/30/3022/A302214.html.
#
#    2.)  If you want high resolution map coastlines you will
#         need to download the appropriate data files, if you
#         have not done so.  For details, see:
#
#            http://www.pyngl.ucar.edu/Graphics/rangs.shtml
#

#
#  Import numpy.
#
import numpy, os

#
#  Import Ngl support functions.
#
import Ngl

#
#  Import Nio for reading netCDF files.
#
import Nio

#
#  Open a netCDF file containing data off the coast of North Carolina.
#
dirc  = Ngl.pynglpath("data")
cfile = Nio.open_file(os.path.join(dirc,"cdf","ctnccl.nc"))

#
#  Read the lat/lon/ele/depth arrays to numpy.arrays.
#
lat   = cfile.variables["lat"][:]
lon   = cfile.variables["lon"][:]
ele   = cfile.variables["ele"][:]
depth = cfile.variables["dat"][:]

#
# Create colormap
#
cmap = numpy.zeros((104,3),'f')
cmap[0] = [1.,1.,1.]
cmap[1] = [0.,0.,0.]
cmap[2] = [.5,.5,.5]
cmap[3] = [.8,.8,.8]

iofc = 151
iolc = 250
for i in xrange(151,251):
  p = (1.*iolc-i)/(1.*iolc-1.*iofc)
  q = (i-1.*iofc)/(1.*iolc-1.*iofc)
  cmap[i-147] = [0.,p,q]

#
#  Open workstation.
#
rlist            = Ngl.Resources()
rlist.wkColorMap = cmap
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"ctnccl",rlist)

#
#  The next set of resources will apply to the contour plot.
#
resources = Ngl.Resources()

resources.nglSpreadColorStart = 4

resources.sfXArray              = lon  # Portion of map on which to overlay
resources.sfYArray              = lat  # contour plot.
resources.sfElementNodes        = ele
resources.sfFirstNodeIndex      = 1

resources.cnFillOn              = True 
resources.cnFillMode            = "RasterFill"
resources.cnRasterSmoothingOn   = True
resources.cnLinesOn             = False
resources.cnLineLabelsOn        = False
resources.cnLevelSelectionMode  = "ExplicitLevels"
resources.cnLevels              = [ 1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,15.,20.,
                                   25.,30.,35.,40.,45.,50.,100.,200.,300.,400.,
                                   500.,600.,700.,800.,900.,1000.,1250.,1500.,
                                   1750.,2000.,2250.,2500.,2750.,3000.,3250.,
                                   3500.,3750.,4000.,4250.,4500.,4750.,5000.]

resources.tiMainString          = "North Carolina Coast (depth in meters)"
resources.tiMainFontHeightF     = 0.015

resources.nglDraw               = False
resources.nglFrame              = False

contour = Ngl.contour(wks,depth,resources)

#
# Retrieve the actual lat/lon end points of the scalar array so
# we know where to overlay on map.
#
xs = Ngl.get_float(contour.sffield,"sfXCActualStartF")
xe = Ngl.get_float(contour.sffield,"sfXCActualEndF")
ys = Ngl.get_float(contour.sffield,"sfYCActualStartF")
ye = Ngl.get_float(contour.sffield,"sfYCActualEndF")

#
#  The next set of resources will apply to the map plot.
#
resources.mpProjection          = "CylindricalEquidistant"

#
# Once the high resolution coastline data files have been
# downloaded (see the Notes section above for details), to
# access them you need to change the following resource 
# to "HighRes".
#
resources.mpDataBaseVersion     = "MediumRes"
resources.mpLimitMode           = "LatLon"
resources.mpMinLonF             = xs
resources.mpMaxLonF             = xe
resources.mpMinLatF             = ys
resources.mpMaxLatF             = ye
resources.mpPerimOn             = True
resources.mpGridAndLimbOn       = False
resources.mpPerimDrawOrder      = "PostDraw"
resources.mpFillDrawOrder       = "PostDraw"
resources.mpFillOn              = True
resources.mpFillColors          = ["background","transparent","LightGray","transparent"]
resources.lbLabelFontHeightF    = 0.01
resources.lbBoxLinesOn          = False
resources.lbOrientation         = "Horizontal"

resources.pmTickMarkDisplayMode    = "Never"
resources.pmLabelBarOrthogonalPosF = -0.05

resources.nglDraw                  = True
resources.nglFrame                 = True

map = Ngl.contour_map(wks,depth,resources)

Ngl.end()
