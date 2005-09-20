#
#  Import NumPy.
#
import Numeric

#
#  Import Ngl support functions.
#
import Ngl

#
#  Import Nio for importing netCDF files.
#
import Nio

#
#  Open a netCDF file containing the grid and data from the HOMME
#  (High-Order Multiscale Modeling Environment) model, formerly
#  called SEAM (Spectral Element Atmosphere Model).
#
dirc  = Ngl.ncargpath("data")
cfile = Nio.open_file(dirc + "/cdf/seam.nc")

#
#  Read the lat/lon/ps arrays to Numeric arrays, and convert
#  to 1D.
#
lon = Numeric.ravel(cfile.variables["lon2d"][:,:])
lat = Numeric.ravel(cfile.variables["lat2d"][:,:])
ps  = Numeric.ravel(cfile.variables["ps"][0,:,:])/100.

#
#  Select a colormap and open a workstation.
#
rlist            = Ngl.Resources()
rlist.wkColorMap = "BlAqGrYeOrReVi200"

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"seam",rlist)

#
#  The next set of resources will apply to the contour plot and the labelbar.
#
resources = Ngl.Resources()

resources.sfXArray            = lon
resources.sfYArray            = lat

resources.nglSpreadColorStart = 176
resources.nglSpreadColorEnd   = 2

resources.cnFillOn            = True
resources.cnFillMode          = "AreaFill"
resources.cnLinesOn           = False
resources.cnLineLabelsOn      = False

resources.lbBoxLinesOn        = False
resources.lbLabelFontHeightF  = 0.015
resources.tiMainString        = "HOMME grid - surface pressure (mb)"

#
# The contour plot is not very interesting, so don't draw it.
# 
resources.nglDraw  = False
resources.nglFrame = False

contour = Ngl.contour(wks,ps,resources)

#
# Retrieve the actual lat/lon end points of the scalar array so
# we know where to overlay on map.
#
xs = Ngl.get_float(contour.sffield,"sfXCActualStartF")
xe = Ngl.get_float(contour.sffield,"sfXCActualEndF")
ys = Ngl.get_float(contour.sffield,"sfYCActualStartF")
ye = Ngl.get_float(contour.sffield,"sfYCActualEndF")

resources.nglDraw           = True        # Turn these resources back on.
resources.nglFrame          = True

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

map = Ngl.contour_map(wks,ps,resources)

resources.cnFillMode      = "RasterFill"
resources.cnMaxLevelCount = 255 

map = Ngl.contour_map(wks,ps,resources)

resources.lbOrientation      = "Horizontal"
#resources.lbLabelFontHeightF = 0.015
resources.mpProjection       = "CylindricalEquidistant"
resources.mpCenterLatF       = 0

map = Ngl.contour_map(wks,ps,resources)

resources.cnRasterSmoothingOn = True
resources.tiMainString        = "HOMME grid: Surface pressure w/smoothing"

map = Ngl.contour_map(wks,ps,resources)

del resources.lbLabelFontHeightF        # Let PyNGL set font height.

resources.lbOrientation = "Vertical"
resources.mpProjection  = "LambertEqualArea"
resources.mpCenterLatF  = 40
resources.mpCenterLonF  = 130
resources.mpPerimOn     = False
resources.lbLabelStride = 15

map = Ngl.contour_map(wks,ps,resources)

Ngl.end()
