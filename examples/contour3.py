#
#  File:
#    contour3.py
#
#  Synopsis:
#    Illustrates drawing dashed negative contours and pointing tickmarks out.
#
#  Categories:
#    Contouring
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    April 2006
#
#  Description:
#    This example shows how to create a contour plot, and then retrieve
#    the contour levels so you can force the negative contours to be
#    dashed. It also shows how to get special longitude/time labels on
#    the X axis, and how to change the labeling style of your contour lines.
#
#  Effects illustrated:
#    o  Using get_float_array to retrieve the contour levels.
#    o  Using set_values to set some new contour resources.
#    o  Pointing tickmarks outward
#    o  Changing the style of the contour line labels
# 
#  Output:
#    This example produces two visualizations:
#      1.)  Contouring with longitude and time labels on X/Y axes, 
#           tickmarks pointing outwards, and negative contours dashed..
#      2.)  Contouring with more contour lines labeled.
#
#  Notes:
#     

import Ngl
import numpy, os

#
#  Import Nio for a NetCDF reader.
#
import Nio

#
# Function for making negative contour lines be dashed.
#
def neg_dash_contours(contour):

#
# Retrieve the contour levels used.
#
  levels = Ngl.get_float_array(contour,"cnLevels")

# 
# Create an array to hold the line dash patterns.
#
  patterns = numpy.zeros((len(levels)),'i')
  patterns[:] = 0     # solid line
#
# Make contour lines above 0 dashed.
#
  for i in xrange(len(levels)):
    if (levels[i] > 0.):
      patterns[i] = 2

  rlist                       = Ngl.Resources()
  rlist.cnLineDashPatterns    = patterns
  rlist.cnMonoLineDashPattern = False
  Ngl.set_values(contour,rlist)

#
# Main program
#
# Open netCDF file and get variables.
#
dirc  = Ngl.pynglpath("data")
cdf_file = Nio.open_file(os.path.join(dirc,"cdf","chi200_ud_smooth.nc"))
chi  = cdf_file.variables["CHI"]
chi  = chi[:,:]/1e6
lon  =  cdf_file.variables["lon"][:]
time =  cdf_file.variables["time"][:]
 
#
# Open a PS file.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"contour3")

#
# Create arrays of longitude/time values and their corresponding labels.
#
lon_values  = Ngl.fspan(0., 315., 8)
lon_labels  = ["0","45E","90E","135E","180","135W","90W","45W"]
time_values = [   0,    30,   61,   89,  120,  150,  181]
time_labels = ["DEC","JAN","FEB","MAR","APR","MAY","JUN"]

cnres = Ngl.Resources()

cnres.nglDraw                  = False    # Turn off draw and frame.
cnres.nglFrame                 = False    # We will do this later.

cnres.sfXArray                 = lon
cnres.sfYArray                 = time

cnres.tmXBMode                 = "Explicit"    # Label X and Y axes
cnres.tmYLMode                 = "Explicit"    # with explicit labels.
cnres.tmXBValues               = lon_values
cnres.tmYLValues               = time_values
cnres.tmXBLabels               = lon_labels
cnres.tmYLLabels               = time_labels

cnres.nglPointTickmarksOutward = True

contour = Ngl.contour(wks,chi,cnres)  # Create, but don't draw contour plot.

neg_dash_contours(contour)           # Dash the negative contours.

Ngl.draw(contour)            # Now draw contours and
Ngl.frame(wks)               # advance the frame.

#
# Set some new resources to control how the lines are labeled.
# Use Ngl.set_values to apply these new resources to our existing
# plot.
#

nrlist                          = Ngl.Resources()

nrlist.cnLineLabelPlacementMode = "Computed"
nrlist.cnLineLabelDensityF      = 3.         # < 1.0 = less, > 1.0 = more
nrlist.cnLineLabelInterval      = 1          # Label every line if possible.
Ngl.set_values(contour,nrlist)

Ngl.draw(contour)            # Draw contours and
Ngl.frame(wks)               # advance the frame.


Ngl.end()
