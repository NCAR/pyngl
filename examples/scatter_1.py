#
#  File:
#    scatter_1.py
#
#  Synopsis:
#    Draws random markers using user-defined markers.
#
#  Category:
#    XY plots
#    polymarkers
#
#  Based on NCL example:
#    scatter_1.ncl
#
#  Author:
#    Karin Meier-Fleischer
#  
#  Date of initial publication:
#    November, 2018
#
#  Description:
#    Basic scatter plot using gsn_y to create an XY plot, and 
#    setting the resource xyMarkLineMode to "Markers" to get 
#    markers instead of lines.
#
#  Effects illustrated:
#    o  Drawing a scatter plot
#    o  Changing the markers in an XY plot
#    o  Changing the marker color in an XY plot
#    o  Changing the marker size in an XY plot
#    o  Generating dummy data using "random_chi"
#
#  Output:
#     A single visualization is produced.     
#
'''
  PyNGL Example: 	scatter_1.py

  -  Adding longitude/latitude labels to a masked Lambert Conformal map
  -  Moving the main title up
  -  Attaching text strings to the outside of a plot
  -  Converting lat/lon values to NDC values
  -  Changing the angle of text strings
  -  Adding a carriage return to a text string using a function code
  
'''
from __future__ import print_function
import numpy as np
import Ngl

#-- generate random data
t = np.random.chisquare(2.0, 50)
np.random.seed(10)                   #-- set a seed for the random number generator

#-- open graphics output
wks = Ngl.open_wks("png","plot_scatter_1")

#-- set resources
res                     =  Ngl.Resources()
res.nglMaximize         =  True                    #-- maximize plot

res.tiMainString        = "Scatter Plot"           #-- add title

res.xyMarkLineMode      = "Markers"                #-- choose to use markers
res.xyMonoMarkLineMode  =  True                    #-- default: only one marker style
res.xyMarkers           =  16                      #-- choose type of marker  
res.xyMarkerColor       = "NavyBlue"               #-- marker color
res.xyMarkerSizeF       =  0.01                    #-- marker size (default 0.01)

#-- draw the plot
plot = Ngl.y(wks,t,res)

Ngl.end()
