import Numeric

#
#  Import all names from the NetCDF module.
#
from Scientific.IO.NetCDF import NetCDFFile

#
#  Import Ngl support functions.
#
import Ngl

# Open netCDF file and read variables
dirc  = Ngl.ncargpath("data")
cfile = NetCDFFile(dirc+"/cdf/trinidad.nc","r")

data = cfile.variables["data"]
lat = cfile.variables["lat"][:]
lon = cfile.variables["lon"][:]
minmax_elevE = cfile.variables["minmax_elevE"][:]
minmax_elevW = cfile.variables["minmax_elevW"][:]
map_cornersW = cfile.variables["map_cornersW"][:]
map_cornersE = cfile.variables["map_cornersE"][:]

min_elev = min(minmax_elevE,minmax_elevW)
max_elev = max(minmax_elevE,minmax_elevW)

cmap = Numeric.array([[1.00, 1.00, 1.00],[0.00, 0.00, 0.00], \
                      [0.51, 0.13, 0.94],[0.00, 0.00, 0.59], \
                      [0.00, 0.00, 0.80],[0.25, 0.41, 0.88], \
                      [0.12, 0.56, 1.00],[0.00, 0.75, 1.00], \
                      [0.63, 0.82, 1.00],[0.82, 0.96, 1.00], \
                      [1.00, 1.00, 0.78],[1.00, 0.88, 0.20], \
                      [1.00, 0.67, 0.00],[1.00, 0.43, 0.00], \
                      [1.00, 0.00, 0.00],[0.78, 0.00, 0.00], \
                      [0.63, 0.14, 0.14],[1.00, 0.41, 0.70]], \
                      Numeric.Float0)

#
#  Select a colormap and open an X11 window.
rlist = Ngl.Resources()
rlist.wkColorMap = cmap
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"topo1",rlist)


res = Ngl.Resources()
res.mpFillOn              = False
res.mpLimitMode           = "Corners"
res.mpDataBaseVersion     = "Ncarg4_1"
res.mpOutlineBoundarySets = "AllBoundaries"
res.mpLeftCornerLonF      = map_cornersW[0] 
res.mpLeftCornerLatF      = map_cornersW[1]
res.mpRightCornerLonF     = map_cornersE[2]
res.mpRightCornerLatF     = map_cornersE[3]

# contour resources
res.sfXArray             = lon
res.sfYArray             = lat
res.cnFillOn             = True
res.cnLinesOn            = False
res.cnRasterModeOn       = True
res.cnLevelSelectionMode = "ExplicitLevels"
res.cnLevels             = [ 5000., 6000., 7000., 8000., 8500., 9000., \
                             9500.,10000.,10500.,11000.,11500.,12000., \
                            12500.,13000.,13500.]

# tickmark resources
res.pmTickMarkDisplayMode = "Always"
res.tmXBLabelFontHeightF  = 0.010

# label resources
res.lbTitleString             = "elevation above mean sea level (feet)"
res.lbTitleFontHeightF        = 0.012
res.lbLabelFontHeightF        = 0.008
res.lbTitleOffsetF            = -0.27
res.lbBoxMinorExtentF         = 0.15
res.pmLabelBarOrthogonalPosF  = -0.01
res.lbOrientation             = "Horizontal"
res.pmLabelBarSide            = "Bottom"

# title resources
res.tiMainString      = "USGS DEM TRINIDAD (1 x 2 degrees)"
res.tiMainFont        = "Helvetica-bold"
res.tiMainOffsetYF    = 0.025
res.tiMainFontHeightF = 0.015

res.nglFrame = False

plot = Ngl.contour_map(wks,data,res)

#
# Draw three text strings afterwards to make sure plot
# gets maximized properly.
#
txres               = Ngl.Resources()
txres.txFontHeightF = 0.015
Ngl.text_ndc(wks,"Min Elevation: 1359", 0.22,0.775,txres)
Ngl.text_ndc(wks,"Scale 1:250,000",     0.50,0.775,txres)
Ngl.text_ndc(wks,"Max Elevation: 4322", 0.85,0.775,txres)

Ngl.frame(wks)

Ngl.end()

