#
#  File:
#    topo1.py
#
#  Synopsis:
#    Draws a topographic map of southern Colorado.
#
#  Category:
#    Contours over maps
#    Labelbar
#    Maps
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    September, 2004
#
#  Description:
#    This example use elevations to draw a raster mode topographic
#    visualization of an area of southern Colorado.
#
#  Effects illustrated:
#    o  Reading from a NetCDF file.
#    o  Setting map resources.
#    o  Setting labelbar resources.
#    o  Contouring in raster mode.
#
#  Output:
#    A single visualization is produced showing a color raster
#    contour topographic map of an area of southern Colorado.
#
#  Notes:
#    The data is from the 1-degree (1:250,000) Digital Elevation
#    Models (DEM) from the US Geological Survey Earth Resources
#    Observation System (EROS) Data Center:
#
#      http://edc.usgs.gov/geodata/
#
#    It is freely available to anyone to download.
#     

import numpy, os

#
#  Import Nio for reading netCDF files.
#
import Nio

#
#  Import Ngl support functions.
#
import Ngl

#
#
# Open netCDF file and read variables
#
dirc  = Ngl.pynglpath("data")
cfile = Nio.open_file(os.path.join(dirc,"cdf","trinidad.nc"))

data         = cfile.variables["data"]
lat          = cfile.variables["lat"][:]
lon          = cfile.variables["lon"][:]
map_cornersW = cfile.variables["map_cornersW"][:]
map_cornersE = cfile.variables["map_cornersE"][:]

cmap = numpy.array([[1.00, 1.00, 1.00],[0.00, 0.00, 0.00], \
                    [0.51, 0.13, 0.94],[0.00, 0.00, 0.59], \
                    [0.00, 0.00, 0.80],[0.25, 0.41, 0.88], \
                    [0.12, 0.56, 1.00],[0.00, 0.75, 1.00], \
                    [0.63, 0.82, 1.00],[0.82, 0.96, 1.00], \
                    [1.00, 1.00, 0.78],[1.00, 0.88, 0.20], \
                    [1.00, 0.67, 0.00],[1.00, 0.43, 0.00], \
                    [1.00, 0.00, 0.00],[0.78, 0.00, 0.00], \
                    [0.63, 0.14, 0.14],[1.00, 0.41, 0.70]], \
                    'f')

#
#  Set the colormap and open a PostScript workstation.
#
rlist            = Ngl.Resources()
rlist.wkColorMap = cmap

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"topo1",rlist)

#
# Variable to hold list of plot resources.
#
res = Ngl.Resources()

#
# Map resources
#
res.mpFillOn              = False
res.mpLimitMode           = "Corners"
res.mpLeftCornerLonF      = map_cornersW[0] 
res.mpLeftCornerLatF      = map_cornersW[1]
res.mpRightCornerLonF     = map_cornersE[2]
res.mpRightCornerLatF     = map_cornersE[3]
res.mpDataBaseVersion     = "MediumRes"
res.mpOutlineBoundarySets = "AllBoundaries"

#
# Contour resources
#
res.sfXArray             = lon
res.sfYArray             = lat
res.cnFillOn             = True
res.cnLinesOn            = False
res.cnFillMode           = "RasterFill"
res.cnLevelSelectionMode = "ExplicitLevels"
res.cnLevels             = [ 5000., 6000., 7000., 8000., 8500., 9000., \
                             9500.,10000.,10500.,11000.,11500.,12000., \
                            12500.,13000.,13500.]
#
# Labelbar resources
#
res.lbTitleString             = "elevation above mean sea level (feet)"
res.lbTitleFontHeightF        = 0.012
res.lbLabelFontHeightF        = 0.008
res.lbTitleOffsetF            = -0.27
res.lbBoxMinorExtentF         = 0.15
res.pmLabelBarOrthogonalPosF  = -0.01
res.lbOrientation             = "Horizontal"

#
# Title resources
#
res.tiMainString      = "USGS DEM TRINIDAD (1 x 2 degrees)"
res.tiMainFont        = "Helvetica-bold"
res.tiMainOffsetYF    = 0.025
res.tiMainFontHeightF = 0.015

res.nglFrame = False

plot = Ngl.contour_map(wks,data,res)

#
# Draw three text strings after the plot is drawn to make sure plot
# gets maximized properly.
#
txres               = Ngl.Resources()
txres.txFontHeightF = 0.015

Ngl.text_ndc(wks,"Min Elevation: 1359", 0.22, 0.775, txres)
Ngl.text_ndc(wks,"Scale 1:250,000",     0.50, 0.775, txres)
Ngl.text_ndc(wks,"Max Elevation: 4322", 0.85, 0.775, txres)

Ngl.frame(wks)

Ngl.end()

