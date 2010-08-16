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
#    Masked arrays
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
#    o Using Ngl.add_new_coord_limits to add new limits to coordinate
#      arrays of an irregular axis.
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
#    This example was updated December 2006 to include frames 4 and 5.
#    It will only work with PyNGL version 1.2.0 or greater.
#

#
#  Import numpy.
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
#  Import masked array module.
#
from numpy import ma

#
#  Open a netCDF file and get some data..
#
data_dir = Ngl.pynglpath("data")
cdf_file = Nio.open_file(os.path.join(data_dir,"cdf","ocean.nc"))
T        = ma.masked_values(cdf_file.variables["T"][:,:],cdf_file.variables["T"]._FillValue)
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
# the top of the plot. Note that because T is a masked array, the
# fill_value of T will be used to fill in the new data array.
#
Tnew,z_t_new = Ngl.add_new_coord_limits(T,ycoord=z_t,ymax=5000)
resources.sfYArray     = z_t_new
resources.tiMainString = "New max limit for Y axis"

plot = Ngl.contour(wks,Tnew,resources)

#
# Set new min/max limits for X and Y axes so we can get some white
# space at the top, right, and left of the plot.
#

Tnew,lat_t_new,z_t_new = Ngl.add_new_coord_limits(T,xcoord=lat_t,
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
