#
#  File:
#    scatter_2.py
#
#  Synopsis:
#    Draws random markers using user-defined markers.
#
#  Category:
#    XY plots
#    polymarkers
#
#  Based on NCL example:
#    scatter_2.ncl
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
#    o  Rotating markers in an XY plot
#    o  Creating your own markers for an XY plot
#    o  Generating dummy data using "random_normal"
#    o  Clipping markers outside the viewport
#
#  Output:
#     Two visualizations are produced.     
#
'''
  PyNGL Example: 	scatter_2.py

  -  Drawing a scatter plot
  -  Changing the markers in an XY plot
  -  Changing the marker color in an XY plot
  -  Changing the marker size in an XY plot
  -  Rotating markers in an XY plot
  -  Creating your own markers for an XY plot
  -  Generating dummy data using "random_normal"
  -  Clipping markers outside the viewport
  
'''
from __future__ import print_function
import numpy as np
import Ngl

#-- generate random data
t = np.random.chisquare(2.0, 50)
np.random.seed(10)                   #-- set a seed for the random number generator

#-- open graphics output
wks = Ngl.open_wks("png","plot_scatter_2")

#-- define new marker
clover = Ngl.new_marker(wks, "p", 35, 0.0, 0.0, 1.3125, 1.5, -45.0)

#-- set resources
res                =  Ngl.Resources()
res.nglMaximize    =  True                    #-- maximize plot

res.tiMainString   = "Make your own marker"   #-- add title

res.xyMarkLineMode = "Markers"                #-- choose to use markers
res.xyMarkers      =  clover                  #-- choose type of marker  
res.xyMarkerColor  = "ForestGreen"            #-- marker color
res.xyMarkerSizeF  =  0.01                    #-- marker size (default 0.01)

#-- first plot
plot = Ngl.y(wks,t,res)

res.vpClipOn       =  True                    #-- clip any markers outside the viewport
res.tiMainString   = "Clipping markers outside the viewport"

#-- second plot
plot = Ngl.y(wks,t,res)                       #-- create and draw plot
  
Ngl.end()
