#
#  Import NumPy.
#
import Numeric

#
#  Import all names from the NetCDF module.
#
from Scientific.IO.NetCDF import *

#
#  Import PyNGL support functions.
#
from Ngl import *
#
#  Open netCDF files.
#
dirc = ncargpath("data")

#
#  Open the netCDF file.
#
ufile = NetCDFFile(dirc + "/cdf/Ustorm.cdf","r")
vfile = NetCDFFile(dirc + "/cdf/Vstorm.cdf","r")
tfile = NetCDFFile(dirc + "/cdf/Tstorm.cdf","r")

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
wks = ngl_open_wks(wks_type,"ngl06p")

#----------- Begin first plot -----------------------------------------

resources = Resources()
if hasattr(u,"_FillValue"):
  resources.vfMissingUValueV = u._FillValue
if hasattr(v,"_FillValue"):
  resources.vfMissingVValueV = v._FillValue

nlon = len(lon)
nlat = len(lat)
resources.vfXCStartV  = lon[0][0]             # Define X/Y axes range
resources.vfXCEndV    = lon[len(lon[:])-1][0] # for vector plot.
resources.vfYCStartV  = lat[0][0]
resources.vfYCEndV    = lat[len(lat[:])-1][0]

map = ngl_vector_map(wks,ua,va,resources)  # Draw a vector plot of u and v

#----------- Begin second plot -----------------------------------------

resources.mpLimitMode       = "Corners"  # Zoom in on the plot area.
resources.mpLeftCornerLonF  = lon[0][0]
resources.mpRightCornerLonF = lon[len(lon[:])-1][0]
resources.mpLeftCornerLatF  = lat[0][0]
resources.mpRightCornerLatF = lat[len(lat[:])-1][0]

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
resources.nglSpreadColors = False    # Do not interpolate color space.

map = ngl_vector_map(wks,ua,va,resources)  # Draw a vector plot.

#----------- Begin third plot -----------------------------------------

tfile = NetCDFFile(dirc+"/cdf/Tstorm.cdf","r")    # Open a netCDF file.
temp = tfile.variables["t"]
tempa = temp[0,:,:]
if hasattr(temp,"_FillValue"):
  tempa = ((tempa-273.15)*9.0/5.0+32.0) *  \
          Numeric.not_equal(tempa,temp._FillValue) + \
          temp._FillValue*Numeric.equal(tempa,temp._FillValue)
  resources.sfMissingValueV = temp._FillValue
else:
  tempa = (tempa-273.15)*9.0/5.0+32.0

temp_units = "(deg F)"

cmap = Numeric.array([[1.00, 1.00, 1.00], [0.00, 0.00, 0.00], \
                      [.560, .500, .700], [.300, .300, .700], \
                      [.100, .100, .700], [.000, .100, .700], \
                      [.000, .300, .700], [.000, .500, .500], \
                      [.000, .700, .100], [.060, .680, .000], \
                      [.550, .550, .000], [.570, .420, .000], \
                      [.700, .285, .000], [.700, .180, .000], \
                      [.870, .050, .000], [1.00, .000, .000], \
                      [.700, .700, .700]],Numeric.Float0)

rlist = Resources()
rlist.wkColorMap = cmap
ngl_set_values(wks,rlist)

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

resources.tiMainString      = ":F25:Wind velocity vectors"  # Title
resources.tiMainFontHeightF = 0.03

resources.pmLabelBarDisplayMode  = "Always"     # Turn on a label bar.
resources.pmLabelBarSide         = "Bottom"     # Change orientation
resources.lbOrientation          = "Horizontal" # of label bar.
resources.lbPerimOn              = False        # Turn off perimeter.
resources.lbTitleString          = "TEMPERATURE (:S:o:N:F)" # Title for
resources.lbTitleFont            = 25                       # label bar.

resources.mpOutlineBoundarySets = "GeophysicalAndUSStates"

map = ngl_vector_scalar_map(wks,ua[::2,::2],va[::2,::2],  \
                            tempa[::2,::2],resources)

del map
del u
del v
del temp
del tempa

ngl_end()
