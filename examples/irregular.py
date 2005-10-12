#
#  File:
#    irregular.py
#
#  Synopsis:
#    Shows how to make irregular axes plot on a linear or log scale.
#
#  Categories:
#    Contouring
#    Special effects
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    December, 2004
#
#  Description:
#    This example shows how to take an axis defined at
#    irregularly spaced coordinates and make that axis
#    in a visualization be drawn on either a linear or
#    log scale; the example also shows how to reverse
#    an axis.
#
#  Effects illustrated:
#    o Using the resources "nglXAxisType" and "nglYAxisType"
#      to specify whether irregularly spaced data along 
#      an X or Y axis should be displayed on a linear or log scale.
#    o Using the resource "trYReverse" to show how to reverse 
#      an axis.
#    o Setting missing values.
# 
#  Output:
#    Five contour visualizations showing:
#      1.) Default settings
#      2.) Y axis displayed on a linear scale
#      3.) X and Y axes displayed on linear scales
#      4.) Y axis reversed
#      5.) Y axis displayed on a log scale
#
#  Notes:
#    This example requires the resource file irregular.res.
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
data_dir = Ngl.pynglpath("data")
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
