#
#  File:
#    color2.py
#
#  Synopsis:
#    Illustrates how to use named colors.
#
#  Category:
#    Colors
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    December, 2005
#
#  Description:
#    Create a color map using named colors, and then use these
#    colors to draw an XY plot with multiple curves.
#
#  Effects illustrated:
#    Using named colors
#    Retrieving resource values.
#    Drawing an XY plot where the X values are just index values.
# 
#  Output:
#    An XY plot is drawn.
#
#  Notes:
#     

import Ngl
import numpy

#
# Create an array of named colors for use as a color map. These
# named colors have predefined RGB values that go with them.
#
# When the color map is retrieved and printed later, you can see 
# what the named colors/RGB pairings are.
#
color_names = ["White","Black","DimGray", "Plum",\
               "CornSilk","NavyBlue","SaddleBrown","Chartreuse", \
               "BurlyWood","DeepPink","Purple","LemonChiffon","DodgerBlue", \
               "Orange","SeaGreen","Gold","Grey11","OliveDrab", \
               "PaleTurquoise","Peru","DarkSalmon","Moccasin","FireBrick"]

rlist = Ngl.Resources()
rlist.wkColorMap = color_names

#
# Open a PS workstation and set this colormap.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"color2",rlist) 

#
# Create four data sets.
#
y  = numpy.zeros([4,21],'f')

pi4 = 3.14159/4.
ptmp = numpy.sin(pi4*numpy.arange(0,21))

for i in range(4):
  y[i,:] = (i+ptmp).astype('f')

#
# Set up a resource list for the XY plot.
#
xyres = Ngl.Resources()

#
# Change line colors and thicknesses.
#
xyres.xyLineColors      = ["Plum","FireBrick","OliveDrab","Orange"]
xyres.xyLineThicknesses = [3.,4.,5.,6.]
xyres.xyMarkers         = [11,12,15,16]
xyres.xyMarkerColors    = ["Grey11","NavyBlue","SaddleBrown","DodgerBlue"]
xyres.xyMarkLineMode    = "MarkLines"

xyres.tiMainFont        = "helvetica-bold"
xyres.tiMainString      = "Using named colors"

#
# Create and draw the XY plot. Note that no X values are being passed
# in here, so index values starting at index 0 will be used.
#
xy = Ngl.y(wks,y,xyres)

#
# Notice when you use Ngl.retrieve_colormap to retrieve the colormap,
# you don't get the named colors back, but rather the RGB
# representations of these colors.
#
cmap = Ngl.retrieve_colormap(wks)

# 
# Print just some of the values of the colormap.
#
print "\n-----------printing some colormap values------------\n"
print "color index  5 (NavyBlue) = ",cmap[5,:]
print "color index 10 (Purple)   = ",cmap[10,:]
print "color index 15 (Gold)     = ",cmap[15,:]

#
# Retrieve some resource values of various types.
#
# Note that in order to retrieve some XY resources, like line colors,
# dash patterns, markers, etc, you need to use the "xydspec" 
# attribute. This represents the DataSpec object which is created
# internally whenever an XY plot is created.
#
cmap_len    = Ngl.get_integer(wks,"wkColorMapLen")
markers     = Ngl.get_integer_array(xy.xydspec,"xyMarkers")
thicknesses = Ngl.get_float_array(xy.xydspec,"xyLineThicknesses")
line_mode   = Ngl.get_string(xy.xydspec,"xyMarkLineMode")
style       = Ngl.get_integer(xy,"xyYStyle")

print "\n-----------printing some resource values------------\n"
print "color map len  = ",cmap_len," (should be 23)"
print "markers        = ",markers," (should be [11 12 15 16])"
print "thicknesses    = ",thicknesses," should be ([3. 4. 5. 6.])"
print "mark/line mode = ",line_mode," (should be 'MarkLines')"
print "style          = ",style," (should be 1)"

#
# Here's another way to retrieve the color map.
#
del cmap
cmap = Ngl.get_MDfloat_array(wks,"wkColorMap")

print "\n-----------printing some more colormap values (float)------------\n"
print "color index  6 (SaddleBrown)   = ",cmap[6,:]
print "color index  7 (Chartreuse)    = ",cmap[7,:]
print "color index 18 (PaleTurquoise) = ",cmap[18,:]

Ngl.end()
