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
from Scientific.IO.NetCDF import *

#
#  Open a netCDF file containing data off the coast of North Carolina.
#
dirc  = Ngl.ncargpath("data")
cfile = NetCDFFile(dirc + "/cdf/ctnccl.nc","r")

#
#  Read the lat/lon/ele/depth arrays to Numeric arrays.
#
lat   = cfile.variables["lat"][:]
lon   = cfile.variables["lon"][:]
ele   = cfile.variables["ele"][:]
depth = cfile.variables["dat"][:]

#
# Create colormap
#
cmap = Numeric.zeros((104,3),Numeric.Float0)
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
# $python_prefx/pythonX.Y/site-packages/PyNGL/ncarg/database/rangs
#
# Now you can change the following resource to "HighRes".
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
