#
# This example shows how to linearize or "logize" an irregular axis,
# using the special resources "nglXAxisType" and "nglYAxisType".
#

#
#  Import NumPy.
#
import Numeric

#
#  Import Nio for reading netCDF files.
#
import Nio

#
#  Import Ngl support functions.
#
import Ngl

#
#  Open a netCDF file and get some data..
#
data_dir = Ngl.ncargpath("data")
cdf_file = Nio.open_file(data_dir + "/cdf/ocean.nc","r")
T        = cdf_file.variables["T"]    
lat_t    = cdf_file.variables["lat_t"][:]
z_t      = cdf_file.variables["z_t"][:] / 100.         # convert cm to m

#
#  Open a workstation.
#
wk_res            = Ngl.Resources()
wk_res.wkColorMap = "gui_default"

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"irregular",wk_res)

#
# Set up a resource list for plot options.
#
resources = Ngl.Resources()

resources.sfXArray        = lat_t
resources.sfYArray        = z_t
resources.sfMissingValueV = T._FillValue[0]

resources.cnFillOn        = True
resources.cnLineLabelsOn  = False

resources.tiMainString    = "Default axes"        # main title

#
# Draw the plot as is, with no transformations in place.
#
plot = Ngl.contour(wks,T,resources)

#
# Linearize the Y axis.
#
resources.tiMainString = "Y axis linearized"
resources.nglYAxisType = "LinearAxis"
plot = Ngl.contour(wks,T,resources)

#
# Linearize the X axis.
#
resources.tiMainString = "X and Y axes linearized"
resources.nglXAxisType = "LinearAxis"
plot = Ngl.contour(wks,T,resources)

#
# Reverse the Y axis.
#
resources.tiMainString = "Y axis reversed"
resources.trYReverse   = True
plot = Ngl.contour(wks,T,resources)

#
# Log the Y axis.
#
resources.tiMainString = "Y axis 'log-ized'"
resources.nglYAxisType = "LogAxis"
plot = Ngl.contour(wks,T,resources)

Ngl.end()
