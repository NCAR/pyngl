#
#  File:
#    scatter2.py
#
#  Synopsis:
#    Draws random markers using user-defined markers.
#
#  Category:
#    XY plots
#    polymarkers
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    December, 2005
#
#  Description:
#    This example generates some random data and plots the data as an
#    XY plot using markers created by the Ngl.new_marker function.
#
#  Effects illustrated:
#      o  Defining your own markers.
#      o  Usage of the the Python "random" module.
#
#  Output:
#     A single visualization is produced showing the random markers
#     on an XY plot.
#
#  Notes:
#     

#
#  Import numpy and random.
#
from __future__ import print_function
import numpy 
import random

#
#  Import Ngl support functions.
#
import Ngl

random.seed(10)   # set a seed for the random number generator

#
# Generate some dummy data.
#
y = numpy.zeros([3,100],'f')
for i in range(100):
  y[0,i] = 90.*random.random()+105.
  y[1,i] = 90.*random.random()+105.
  y[2,i] = 90.*random.random()+105.

wks_type = "png"
wks = Ngl.open_wks(wks_type,"scatter2")

#
# Set up parameters for creating some new markers.
#
mrk_indices = numpy.zeros(3,'i')

mstrings = ["u","z","y"]     # triangle, star, sqaure
fontnums = [34,35,35]
yoffsets = [0.4, 0.0, 0.0]
sizes    = [2.0, 1.5, 1.0]
mrk_indices[0] = Ngl.new_marker(wks, mstrings[0], fontnums[0], 0, \
                                yoffsets[0], 1, sizes[0], 15.)
mrk_indices[1] = Ngl.new_marker(wks, mstrings[1], fontnums[1], 0, \
                                yoffsets[1], 1, sizes[1], 0.)
mrk_indices[2] = Ngl.new_marker(wks, mstrings[2], fontnums[2], 0, \
                                yoffsets[2], 1, sizes[2], 0.)

#
# Set up resource list for XY plot.
#
res                     = Ngl.Resources()
res.xyMarkLineMode      = "Markers"        # Default is to draw lines.
res.xyMonoMarkLineMode  = True             # Default is only one marker style.
res.xyMarkers           = mrk_indices      # Set new markers
res.xyMarkerColors      = ["red","green","blue"]
res.tiMainString        = "Scatter plot with user-defined markers"
  
plot = Ngl.y(wks,y,res)    # Draw the plot.

Ngl.end()
