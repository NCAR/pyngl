#
#  File:
#    scatter_5.py
#
#  Synopsis:
#    Demonstrates how to take a 1D array of data, and group the values so you can 
#    mark each group with a different marker and color using Ngl.y.
#
#  Category:
#    xy plots
#    overlay
#
#  Based on NCL example:
#    scatter_5.ncl
#
#  Author:
#    Karin Meier-Fleischer
#  
#  Date of initial publication:
#    November, 2018
#
#  Description:
#    This example shows how to use different markers in a scatter plot.
#
#  Effects illustrated:
#    o  Drawing a scatter plot with markers of different colors
#    o  Generating dummy data using "random_chi"
#    o  Drawing a legend outside an XY plot
#    o  Changing the markers in an XY plot
#    o  Changing the marker color in an XY plot
#    o  Changing the marker size in an XY plot
#    o  Manually creating a legend using markers and text
#    o  Adding text to a plot
#    o  Using "nice_mnmxintvl" to select a nice span of values through the data
#    o  Creating a color map using named colors
#    o  Moving a legend closer to a plot
#    o  Customizing the labels in a legend
#    o  Changing the orientation of a legend
#
#  Output:
#     A single visualization is produced.     
#
'''
  PyNGL Example: 	scatter_5.py

  -  Drawing a scatter plot with markers of different colors
  -  Generating dummy data using "random_chi"
  -  Drawing a legend outside an XY plot
  -  Changing the markers in an XY plot
  -  Changing the marker color in an XY plot
  -  Changing the marker size in an XY plot
  -  Manually creating a legend using markers and text
  -  Adding text to a plot
  -  Using "nice_mnmxintvl" to select a nice span of values through the data
  -  Creating a color map using named colors
  -  Moving a legend closer to a plot
  -  Customizing the labels in a legend
  -  Changing the orientation of a legend
  
'''
from __future__ import print_function
import numpy as np
import Ngl,Nio

#-- generate random data 
npts   = 300
low    = 0
high   = 50
lint   = 10

data1d = np.random.randint(low=low,high=high,size=npts)

#-- create roughly ten equally spaced levels through the data
levels  = range(low,high+lint,lint)
nlevels = len(levels)

print(levels)
print(nlevels)

#-- assign 2D array
data2d = np.ma.zeros([nlevels-1,npts],dtype=float)

print(data1d)
print("---------------------------------------------------------")

#-- select data of levels
fillvalue = -9999.9
for i in range(0,nlevels-1):
   data2d[i,:] = np.where((data1d >= levels[i]) & (data1d < levels[i+1]), data1d, fillvalue)
   data2d[i,:] = np.ma.masked_where(data2d[i,:] == fillvalue, data2d[i,:])

print("---------------------------------------------------------")
print(data2d)

#-- group the values and put in 2D array
labels = ["" for x in range(nlevels)]
for i in range(0,nlevels-1):
  ii = np.argwhere((data1d >= levels[i]) & (data1d < levels[i+1]))
  data2d[i,ii] = data1d[ii]
  labels[i] = str(levels[i]) + ":" + str(levels[i+1])
  del(ii)

#-----------------------------------------------------------
# Graphics
#-----------------------------------------------------------
wks = Ngl.open_wks("png","plot_scatter_5")

#-- define the colors
colors = ["white","black","darkgoldenrod","darkgreen","coral4",\
          "cyan3","firebrick1","darkslateblue","limegreen",\
          "lightgoldenrod","darkseagreen1","lightsteelblue1"]
Ngl.define_colormap(wks,colors)                 #-- set colormap of workstation

#-- define color and marker index lists
cols    = list(range(2,12))  
markers = list(range(2,16))
          
#-- plot resources
res                        =  Ngl.Resources()
res.nglFrame               =  False
res.nglMaximize            =  True              #-- maximize plot
       
res.tiMainString           = "Scatter plot with grouped markers" #-- add title

res.lgOrientation          = "horizontal"       #-- default is vertical
res.lgPerimOn              =  False             #-- turn off the perimeter box
res.lgLabelFontHeightF     =  0.008

res.pmLegendDisplayMode    = "Always"           #-- turn on the legend
res.pmLegendWidthF         =  0.75              #-- make it wider

res.xyExplicitLabels       =  labels            #-- set the labels
res.xyMarkLineMode        = ["Markers"]        #-- use markers
res.xyMonoMarkLineMode     =  False
res.xyMonoMarker           =  False
res.xyMarkers              =  markers           #-- again, you can list more than you need
res.xyMarkerThicknessF     =  2.5
res.xyMarkerColors         =  colors[2:]              #-- it's okay to list more than you need here

plot  = Ngl.y(wks,data2d,res)

Ngl.frame(wks)

Ngl.end()


