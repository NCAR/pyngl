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
cfile = NetCDFFile(dirc + "/cdf/seam.nc")

#
#  Read the lat/lon/ps arrays to Numeric arrays, and convert
#  to 1D.
#
lon = Numeric.ravel(cfile.variables["lon2d"][:,:])
lat = Numeric.ravel(cfile.variables["lat2d"][:,:])
ps  = Numeric.ravel(cfile.variables["ps"][0,:,:])

#
#  Select a colormap and open a workstation.
#
rlist            = Ngl.Resources()
rlist.wkColorMap = "BlAqGrYeOrReVi200"

wks_type = "x11"
wks = Ngl.open_wks(wks_type,"seam",rlist)

#
#  The next set of resources will apply to the contour plot and the labelbar.
#
resources = Ngl.Resources()

resources.sfXArray          = lon
resources.sfYArray          = lat

resources.nglSpreadColorStart = 176
resources.nglSpreadColorEnd   = 2

resources.cnFillOn              = True
resources.cnFillMode            = "AreaFill"
resources.cnLinesOn             = False
resources.cnLineLabelsOn        = False

#resources.lbOrientation         = "Horizontal"
resources.lbLabelAutoStride     = True
resources.lbBoxLinesOn          = False
resources.lbLabelFontHeightF    = 0.02
resources.lbAutoManage          = False
#resources.pmLabelBarSide        = "Bottom"
#resources.pmLabelBarDisplayMode = "Never"
resources.tiMainString          = "SEAM grid - surface pressure"

resources.nglDraw = False
resources.nglFrame = False
contour = Ngl.contour(wks,ps,resources)
resources.nglDraw = True
resources.nglFrame = True

#
# Retrieve the actual lat/lon end points of the scalar array so
# we know where to overlay on map.
#
xs = Ngl.get_float(contour.sffield,"sfXCActualStartF")
xe = Ngl.get_float(contour.sffield,"sfXCActualEndF")
ys = Ngl.get_float(contour.sffield,"sfYCActualStartF")
ye = Ngl.get_float(contour.sffield,"sfYCActualEndF")

resources.mpProjection      = "Orthographic"
resources.mpDataBaseVersion = "MediumRes"
resources.mpLimitMode       = "LatLon"
resources.mpMinLonF         = xs
resources.mpMaxLonF         = xe
resources.mpMinLatF         = ys
resources.mpMaxLatF         = ye
resources.mpPerimOn         = False
resources.mpCenterLatF      =  40
resources.mpCenterLonF      = -130

resources.pmTickMarkDisplayMode = "Always"

#map = Ngl.contour_map(wks,ps,resources)

resources.cnFillMode      = "RasterFill"
resources.cnMaxLevelCount = 255 

#map = Ngl.contour_map(wks,ps,resources)

resources.mpProjection = "CylindricalEquidistant"
resources.mpCenterLatF = 0

#map = Ngl.contour_map(wks,ps,resources)

resources.cnRasterSmoothingOn = True
resources.tiMainString        = "Surface pressure with smoothing on" 

#map = Ngl.contour_map(wks,ps,resources)

resources.tiMainString = "SEAM grid: Surface pressure w/smoothing"
resources.mpProjection = "LambertEqualArea"
resources.mpCenterLatF = 40
resources.mpCenterLonF = 130
resources.mpPerimOn  = False
resources.lbLabelStride = 15
resources.lbLabelAutoStride = False
map = Ngl.contour_map(wks,ps,resources)

Ngl.end()
