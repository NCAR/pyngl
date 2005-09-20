#
#  Import NumPy.
#
import Numeric

#
#  Import Nio read functions (for netCDF files).
#
import Nio

#
#  To use the ScientificPython module to read in the netCDF file,
#  comment out the above "import Nio" command, and uncomment the 
#  import line below.
#
# from Scientific.IO.NetCDF import NetCDFFile

#
#  Import Ngl support functions.
#
import Ngl
#
#  Open the netCDF files.
#
dirc  = Ngl.ncargpath("data")
ufile = Nio.open_file(dirc + "/cdf/Ustorm.cdf","r")  # Open two netCDF files.
vfile = Nio.open_file(dirc + "/cdf/Vstorm.cdf","r")

#
#  This is the ScientificPython method for opening netCDF files.
#
# ufile = NetCDFFile(dirc + "/cdf/Ustorm.cdf","r")  # Open two netCDF files.
# vfile = NetCDFFile(dirc + "/cdf/Vstorm.cdf","r")

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
wks = Ngl.open_wks(wks_type,"ngl03p")

resources = Ngl.Resources()
if hasattr(u,"_FillValue"):
  resources.vfMissingUValueV = u._FillValue
if hasattr(v,"_FillValue"):
  resources.vfMissingVValueV = v._FillValue

vc = Ngl.vector(wks,ua,va,resources)

#----------- Begin second plot -----------------------------------------

resources.vcMinFracLengthF = 0.33
resources.vcRefMagnitudeF  = 20.0
resources.vcRefLengthF     = 0.045
resources.vcMonoLineArrowColor  = False   # Draw vectors in color.
resources.nglSpreadColors = False    # Do not interpolate color space.

vc = Ngl.vector(wks,ua,va,resources)

#----------- Begin third plot -----------------------------------------

resources.tiMainString  = ":F26:wind velocity vectors - January 1996"
resources.tiXAxisString = "longitude"
resources.tiYAxisString = "latitude"
 
resources.vfXCStartV  = lon[0][0]            # Define X/Y axes range
resources.vfXCEndV    = lon[len(lon[:])-1][0] # for vector plot.
resources.vfYCStartV  = lat[0][0]
resources.vfYCEndV    = lat[len(lat[:])-1][0]

resources.pmLabelBarWidthF         = 0.1

vc = Ngl.vector(wks,ua,va,resources)

#---------- Begin fourth plot ------------------------------------------

tfile = Nio.open_file(dirc+"/cdf/Tstorm.cdf","r")    # Open a netCDF file.
#
#  This is the ScientificPython method for opening netCDF files.
#
# tfile = NetCDFFile(dirc+"/cdf/Tstorm.cdf","r")    # Open a netCDF file.

temp = tfile.variables["t"]

tempa = temp[0,:,:]
#
#  Convert from degrees Kelvin to degrees F.
#  If the data has a fill value, do the conversion  at those 
#  values not equal to the fill value and fill in with the fill 
#  value elsewhere.
#
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

rlist = Ngl.Resources()
rlist.wkColorMap = cmap
Ngl.set_values(wks,rlist)

resources.vcFillArrowsOn           = True  # Fill the vector arrows
resources.vcMonoFillArrowFillColor = False # in different colors
resources.vcFillArrowEdgeColor     = 1     # Draw the edges in black.
resources.vcFillArrowWidthF        = 0.055 # Make vectors thinner.

resources.tiMainString      = ":F26:wind velocity vectors colored by temperature " + temp_units
resources.tiMainFontHeightF = 0.02  # Make font slightly smaller.
resources.lbLabelFont       = 21    # Change font of label bar labels.

vc = Ngl.vector_scalar(wks,ua,va,tempa,resources) # Draw a vector plot of

del vc
del u
del v
del temp
del tempa

Ngl.end()
