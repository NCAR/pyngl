#
#  Import Python modules to be used.
#
import Numeric,sys,os

#
#  Import all names from the NetCDF module.
#
from Scientific.IO.NetCDF import *

#
#  Import Ngl support functions.
#
from Ngl import *

#  
#  Open the netCDF file.
#  
cdf_file = NetCDFFile(ncargpath("data") + "/cdf/contour.cdf","r")

#
#  Associate Python variables with NetCDF variables.
#  These variables have associated attributes.
#
temp = cdf_file.variables["T"]    # temperature
Z    = cdf_file.variables["Z"]    # geopotential height
pres = cdf_file.variables["Psl"]  # pressure at mean sea level
lat  = cdf_file.variables["lat"]  # latitude
lon  = cdf_file.variables["lon"]  # longitude

#
#  Open a workstation.
#
wks = ngl_open_wks("x11","ngl02p")

resources = Resources()

#
#  Define a Numeric data array containing the temperature
#  for the first time step and first level.
#  This array does not have the attributes that are
#  associated with the variable temp.
#
tempa = temp[0,0,:,:]

#
#  Convert tempa from Kelvin to Celcius while retaining the missing values.
#  If you were to write the new temp data to a netCDF file, then you
#  would want to change the temp units to "(C)".
#
if hasattr(temp,"_FillValue"):
  tempa = ((tempa-273.15)*Numeric.not_equal(tempa,temp._FillValue)) +   \
        temp._FillValue*Numeric.equal(tempa,temp._FillValue)
else:
  tempa = tempa - 273.15

#
#  Set the scalarfield missing value if temp has one specified.
#
if hasattr(temp,"_FillValue"):
  resources.sfMissingValueV = temp._FillValue[0]

#
#  Specify the main title base on the long_name attribute of temp.
#
if hasattr(temp,"long_name"):
  resources.tiMainString = temp.long_name

plot = ngl_contour(wks,tempa,resources)

#----------- Begin second plot -----------------------------------------

resources.cnMonoLineColor = False  # Allow multiple colors for contour lines.
resources.tiMainString    = "Temperature (C)"

plot = ngl_contour(wks,tempa,resources)  # Draw a contour plot.

#----------- Begin third plot -----------------------------------------

resources.cnFillOn          = True    # Turn on contour line fill.
resources.cnMonoFillPattern = False   # Turn off using a single fill pattern.
resources.cnMonoFillColor   = True
resources.cnMonoLineColor   = True

if hasattr(lon,"long_name"):
  resources.tiXAxisString = lon.long_name
if hasattr(lat,"long_name"):
  resources.tiYAxisString = lat.long_name
resources.sfXArray        = lon[:]
resources.sfYArray        = lat[:]
resources.pmLabelBarDisplayMode = "Never" # Turn off label bar.

plot = ngl_contour(wks,tempa,resources)   # Draw a contour plot.

#---------- Begin fourth plot ------------------------------------------

resources.cnMonoFillPattern     = True     # Turn solid fill back on.
resources.cnMonoFillColor       = False    # Use multiple colors.
resources.cnLineLabelsOn        = False    # Turn off line labels.
resources.cnInfoLabelOn         = False    # Turn off informational
                                             # label.
resources.cnLinesOn             = False    # Turn off contour lines.
resources.nglSpreadColors         = False    # Do not interpolate color space.

resources.pmLabelBarDisplayMode = "Always" # Turn on label bar.
resources.lbPerimOn             = False    # Turn off perimeter on
                                             # label bar.

resources.tiMainFont      = 26
resources.tiXAxisFont     = 26
resources.tiYAxisFont     = 26

if hasattr(Z,"_FillValue"):
  resources.sfMissingValueV = Z._FillValue
if hasattr(Z,"long_name"):
  resources.tiMainString = Z.long_name
plot = ngl_contour(wks,Z[0,0,:,:],resources)    # Draw a contour plot.

#---------- Begin fifth plot ------------------------------------------

cmap = Numeric.array([[0.00, 0.00, 0.00], [1.00, 1.00, 1.00], \
                      [0.10, 0.10, 0.10], [0.15, 0.15, 0.15], \
                      [0.20, 0.20, 0.20], [0.25, 0.25, 0.25], \
                      [0.30, 0.30, 0.30], [0.35, 0.35, 0.35], \
                      [0.40, 0.40, 0.40], [0.45, 0.45, 0.45], \
                      [0.50, 0.50, 0.50], [0.55, 0.55, 0.55], \
                      [0.60, 0.60, 0.60], [0.65, 0.65, 0.65], \
                      [0.70, 0.70, 0.70], [0.75, 0.75, 0.75], \
                      [0.80, 0.80, 0.80], [0.85, 0.85, 0.85]],Numeric.Float0)

#
#  Specify a new color map.
#
rlist = Resources()
rlist.wkColorMap = cmap
ngl_set_values(wks,rlist)

#
#  If the pressure field has a long_name attribute, use it for a title.
#
if hasattr(pres,"long_name"):
  resources.tiMainString = pres.long_name

#
#  Convert the pressure to millibars while retaining the missing values.
#
presa = pres[0,:,:]
if hasattr(pres,"_FillValue"):
  presa = (0.01*presa*Numeric.not_equal(presa,pres._FillValue)) +   \
        pres._FillValue*Numeric.equal(presa,pres._FillValue)
else:
  presa = 0.01*presa

plot = ngl_contour(wks,presa,resources)  # Draw a contour plot.

print "\nSubset [2:6,7:10] of temp array:" # Print subset of "temp" variable.
print tempa[2:6,7:10]
print "\nDimensions of temp array:"        # Print dimension names of T.
print temp.dimensions
print "\nThe long_name attribute of T:"    # Print the long_name attribute of T.
print temp.long_name 
print "\nThe nlat data:"                   # Print the lat data.
print lat[:]           
print "\nThe nlon data:"                   # Print the lon data.
print lon[:]          

#
#  Write a subsection of tempa to an ASCII file.
#
os.system("/bin/rm -f data.asc")
sys.stdout = open("data.asc","w")
for i in range(7,2,-2):
  for j in range(0,5):
    print "%9.5f" % (tempa[i,j])

# Clean up (not really necessary, but a good practice).

del plot 
del resources
del temp

ngl_end()
