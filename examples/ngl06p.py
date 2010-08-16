#
#  File:
#    ngl06p.py
#
#  Synopsis:
#    Draws vectors over maps.
#
#  Category:
#    Vectors over maps.
#
#  Author:
#    Fred Clare (based on a code of Mary Haley).
#  
#  Date of initial publication:
#    September, 2004
#
#  Description:
#    This example draws vectors over maps illustrating
#    various vector resource settings and using a couple
#    of different map projections.
#
#  Effects illustrated:
#    o  Drawing vectors over specified map regions.
#    o  Drawing colored vectors.
#    o  Specifying vector sizes.
#    o  Using filled vectors.
#    o  Drawing a grid only over the oceans.
#    o  Putting a horizontal label bar at the bottom of the plot.
# 
#  Output:
#    This example produces three visualizations:
#      1.)  Monocolored vectors over the U.S. 
#      2.)  Colored vectors.
#      3.)  Filled vectors over the U.S. with label bar at bottom.
#
#  Notes:
#     

#
#  Import numpy.
#
import numpy, os

#
#  Import Nio for a NetCDF reader.
#
import Nio

#
#  To use the ScientificPython module to read in the netCDF file,
#  comment out the above "import" command, and uncomment the 
#  import line below.
#
# from Scientific.IO.NetCDF import NetCDFFile

#
#  Import PyNGL support functions.
#
import Ngl
#
#  Open netCDF files.
#
dirc = Ngl.pynglpath("data")

#
#  Open the netCDF files.
#
ufile = Nio.open_file(os.path.join(dirc,"cdf","Ustorm.cdf"),"r")
vfile = Nio.open_file(os.path.join(dirc,"cdf","Vstorm.cdf"),"r")
tfile = Nio.open_file(os.path.join(dirc,"cdf","Tstorm.cdf"),"r")

#
#  This is the ScientificPython method for opening netCDF files.
#
# ufile = NetCDFFile(os.path.join(dirc,"cdf","Ustorm.cdf"),"r")
# vfile = NetCDFFile(os.path.join(dirc,"cdf","Vstorm.cdf"),"r")
# tfile = NetCDFFile(os.path.join(dirc,"cdf","Tstorm.cdf"),"r")

#
#  Get the u/v variables.
#
u = ufile.variables["u"]
v = vfile.variables["v"]
lat = ufile.variables["lat"]
lon = ufile.variables["lon"]
ua = u[0,:,:]
va = v[0,:,:]

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"ngl06p")

#----------- Begin first plot -----------------------------------------

resources = Ngl.Resources()

nlon = len(lon)
nlat = len(lat)
resources.vfXCStartV  = float(lon[0])             # Define X/Y axes range
resources.vfXCEndV    = float(lon[len(lon[:])-1]) # for vector plot.
resources.vfYCStartV  = float(lat[0])
resources.vfYCEndV    = float(lat[len(lat[:])-1])

map = Ngl.vector_map(wks,ua,va,resources)  # Draw a vector plot of u and v

#----------- Begin second plot -----------------------------------------

cmap = numpy.array([[1.00, 1.00, 1.00], [0.00, 0.00, 0.00], \
                    [.560, .500, .700], [.300, .300, .700], \
                    [.100, .100, .700], [.000, .100, .700], \
                    [.000, .300, .700], [.000, .500, .500], \
                    [.000, .700, .100], [.060, .680, .000], \
                    [.550, .550, .000], [.570, .420, .000], \
                    [.700, .285, .000], [.700, .180, .000], \
                    [.870, .050, .000], [1.00, .000, .000], \
                    [.700, .700, .700]],'f')

rlist = Ngl.Resources()
rlist.wkColorMap = cmap
Ngl.set_values(wks,rlist)

resources.mpLimitMode       = "Corners"  # Zoom in on the plot area.
resources.mpLeftCornerLonF  = float(lon[0])
resources.mpRightCornerLonF = float(lon[len(lon[:])-1])
resources.mpLeftCornerLatF  = float(lat[0])
resources.mpRightCornerLatF = float(lat[len(lat[:])-1])

resources.mpPerimOn         =   True    # Turn on map perimeter.

resources.vpXF      = 0.1   # Increase size and change location
resources.vpYF      = 0.92  # of vector plot.
resources.vpWidthF  = 0.75
resources.vpHeightF = 0.75

resources.vcMonoLineArrowColor = False  # Draw vectors in color.
resources.vcMinFracLengthF     = 0.33   # Increase length of
resources.vcMinMagnitudeF      = 0.001  # vectors.
resources.vcRefLengthF         = 0.045
resources.vcRefMagnitudeF      = 20.0

map = Ngl.vector_map(wks,ua,va,resources)  # Draw a vector plot.

#----------- Begin third plot -----------------------------------------

temp       = tfile.variables["t"]
tempa      = (temp[0,:,:]-273.15)*9.0/5.0+32.0
temp_units = "(deg F)"

resources.mpProjection = "Mercator"  # Change the map projection.
resources.mpCenterLonF = -100.0
resources.mpCenterLatF =   40.0

resources.mpLimitMode  = "LatLon"  # Change the area of the map
resources.mpMinLatF    =  18.0     # viewed.
resources.mpMaxLatF    =  65.0
resources.mpMinLonF    = -128.
resources.mpMaxLonF    = -58.

resources.mpFillOn               = True  # Turn on map fill.
resources.mpLandFillColor        = 16    # Change land color to grey.
resources.mpOceanFillColor       = -1    # Change oceans and inland
resources.mpInlandWaterFillColor = -1    # waters to transparent.

resources.mpGridMaskMode         = "MaskNotOcean"  # Draw grid over ocean.
resources.mpGridLineDashPattern  = 2
resources.vcFillArrowsOn           = True  # Fill the vector arrows
resources.vcMonoFillArrowFillColor = False # using multiple colors.
resources.vcFillArrowEdgeColor     = 1     # Draw the edges in black.

resources.tiMainString      = "~F25~Wind velocity vectors"  # Title
resources.tiMainFontHeightF = 0.03

resources.pmLabelBarDisplayMode  = "Always"       # Turn on a label bar.
resources.pmLabelBarSide         = "Bottom"       # Change orientation
resources.lbOrientation          = "Horizontal"   # Orientation of label bar.
resources.lbTitleString          = "TEMPERATURE (~S~o~N~F)"

resources.mpOutlineBoundarySets = "GeophysicalAndUSStates"

map = Ngl.vector_scalar_map(wks,ua[::2,::2],va[::2,::2],  \
                            tempa[::2,::2],resources)

del map
del u
del v
del temp
del tempa

Ngl.end()
