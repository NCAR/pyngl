#
#  File:
#    irregular.py
#
#  Synopsis:
#    Shows how to make irregular axes plot on a linear or log scale and
#    how to change axis limits for irregular axes.
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
#    log scale; the example also shows how to change the
#    limits of irregular axes, and how to reverse an axis.
#
#  Effects illustrated:
#    o Using the resources "nglXAxisType" and "nglYAxisType"
#      to specify whether irregularly spaced data along 
#      an X or Y axis should be displayed on a linear or log scale.
#    o Using the resource "trYReverse" to show how to reverse 
#      an axis.
#    o Setting missing values.
#    o Using Ngl.change_coord_limits to change limits of an irregular axis.
# 
#  Output:
#    Seven contour visualizations showing:
#      1.) Default settings
#      2.) Y axis displayed on a linear scale
#      3.) X and Y axes displayed on linear scales
#      4.) Y axis limit increased.
#      5.) X/Y axes limits changed.
#      6.) Y axis reversed
#      7.) Y axis displayed on a log scale
#
#  Notes:
#

#
#  Import Numeric.
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
# In order to change the axes limits for the case where sfXArray and/or
# sfYArray are being set to arrays, you need to create a new data array
# that has one extra element on the axis side you want to change the
# limit for.
#
# This function checks which axes limits are to be changed, and creates
# new data and coord arrays with the extra element(s) added. The data
# array must have a missing value associated with it.


#
#  Open a netCDF file and get some data..
#
data_dir = Ngl.pynglpath("data")
cdf_file = Nio.open_file(data_dir + "/cdf/ocean.nc","r")
T        = cdf_file.variables["T"][:,:]
Tmsg     = cdf_file.variables["T"]._FillValue
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
resources.sfMissingValueV = Tmsg

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
# Set new max limit for Y axis so we can get some white space at
# the top of the plot.
#
Tnew,z_t_new = Ngl.change_coord_limits(T,Tmsg,ycoord=z_t,ymax=5000)
resources.sfYArray     = z_t_new
resources.tiMainString = "New max limit for Y axis"

plot = Ngl.contour(wks,Tnew,resources)

#
# Set new min/max limits for X and Y axes so we can get some white
# space at the top, right, and left of the plot.
#

Tnew,lat_t_new,z_t_new = Ngl.change_coord_limits(T,Tmsg,xcoord=lat_t,
                                                 ycoord=z_t,
                                                 ymin=0,ymax=5000,
                                                 xmin=-40,xmax=40)
resources.sfYArray     = z_t_new
resources.sfXArray     = lat_t_new
resources.tiMainString = "New min/max limits for X and Y axes"

plot = Ngl.contour(wks,Tnew,resources)

#
# Reverse the Y axis.
#
resources.tiMainString = "Y axis reversed"
resources.trYReverse   = True
plot = Ngl.contour(wks,Tnew,resources)

#
# Log the Y axis (and go back to original X/Y axes limits).
#
resources.sfYArray     = z_t
resources.sfXArray     = lat_t
resources.tiMainString = "Y axis 'log-ized'"
resources.nglYAxisType = "LogAxis"
plot = Ngl.contour(wks,T,resources)

Ngl.end()
