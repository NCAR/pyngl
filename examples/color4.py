#
#  File:
#    color4.py
#
#  Synopsis:
#    Draws sixteen sample color boxs with RGB labels.
#
#  Category:
#    Colors
#
#  Author:
#    Fred Clare
#
#  Date of initial publication:
#    January, 2006
#
#  Description:
#    This example draws sixteen color boxes using the RGB
#    values for named colors.  The boxes are labeled with
#    the color name and the associated RGB values.
#
#  Effects illustrated:
#    o  Drawing lines and polygons in NDC space.
#    o  RGB equivalents for some named colors.
#    o  Converting integer RGB color specifications to floating point.
#
#  Output:
#    o One plot is produced with sixteen sample color boxes.
#

import Ngl
import numpy

#
#  Define the colors and labels to be used.
#
colors_and_labels = \
     [                                  \
       [233, 150, 122],  "DarkSalmon",  \
       [164,  42,  42],  "Brown",       \
       [255, 127,   0],  "DarkOrange1", \
       [255,   0,   0],  "Red",         \
       [255, 255,   0],  "Yellow",      \
       [  0, 255,   0],  "Green",       \
       [ 34, 139,  34],  "ForestGreen", \
       [  0, 255, 255],  "Cyan",        \
       [ 79, 148, 205],  "SteelBlue3",  \
       [  0,   0, 255],  "Blue",        \
       [148,   0, 211],  "DarkViolet",  \
       [255,   0, 255],  "Magneta",     \
       [255, 255, 255],  "White",       \
       [153, 153, 153],  "Gray60",      \
       [102, 102, 102],  "Gray40",      \
       [  0,   0,   0],  "Black"        \
     ]

#
#  Open a workstation with a default color table having
#  background color "black" and foreground color "white".
#
rlist = Ngl.Resources()
rlist.wkColorMap = "default"
rlist.wkForegroundColor = "White"
rlist.wkBackgroundColor = "Black"
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"color4",rlist) 

#
#  Extract the colors and labels.
#
colors = colors_and_labels[0:len(colors_and_labels):2]
labels = colors_and_labels[1:len(colors_and_labels):2]

#
#  Set up arrays and resource lists for drawing the boxes.
#  Select "Helvetica-Bold" for all text.
#
x = numpy.zeros(5,'f')
y = numpy.zeros(5,'f')
poly_res = Ngl.Resources()
text_res = Ngl.Resources()
text_res.txFont = "Helvetica-Bold"

#
#  Draw the color boxes and titles.
#
for i in xrange(0,len(colors)):
#
#  delx_0 - horizontal spacing between boxes.
#  delx_1 - width of a box.
#  dely_0 - vertical spacing between boxes.
#  dely_1 - height of a box.
#
  delx_0, delx_1, dely_0, dely_1 = 0.245, 0.235, 0.22, 0.15

  x[0], y[0] = 0.015 + delx_0*(i%4), 0.90 - (i/4)*dely_0 
  x[1], y[1] = x[0] + delx_1       , y[0]
  x[2], y[2] = x[1]                , y[1] - dely_1
  x[3], y[3] = x[0]                , y[2]
  x[4], y[4] = x[0]                , y[0]

#
#  Convert the integer color values obtained from the
#  named color chart (as entered above) to floating 
#  point numbers in the range 0. to 1. (as required 
#  by "Ngl.new_color").
#
  r, g, b = colors[i][0]/255., colors[i][1]/255., colors[i][2]/255.
  poly_res.gsFillColor = Ngl.new_color(wks, r, g, b)
#
#  Draw a white outline if the color is black, otherwise draw a colored box.
#
  if (labels[i] == "Black"):
    Ngl.polyline_ndc(wks, x, y, poly_res)
  else:
    Ngl.polygon_ndc(wks, x, y, poly_res)
#
#  Label the boxes.
#
  text_res.txFontHeightF = 0.017
  Ngl.text_ndc(wks, labels[i], 0.5*(x[0]+x[1]), y[0] + 0.0125, text_res)
  rgb_label = "R=%4.2f G=%4.2f B=%4.2f" % (r, g, b)
  text_res.txFontHeightF = 0.015
  Ngl.text_ndc(wks, rgb_label, 0.5*(x[0]+x[1]), y[3] - 0.0125, text_res)

#
#  Plot top and bottom labels.
#
text_res.txFontHeightF = 0.025
Ngl.text_ndc(wks, "Sixteen Sample Colors", 0.5, 0.96, text_res)
text_res.txFontHeightF = 0.018
Ngl.text_ndc(wks, "The titles below each box indicate Red, Green, and Blue intensity values.", 0.5, 0.035, text_res)
      
Ngl.frame(wks)
Ngl.end()
