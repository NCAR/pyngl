#
#  File:
#    color3.py
#
#  Synopsis:
#    Draws HSV color wedges and tests color conversion functions.
#
#  Category:
#    Colors
#
#  Author:
#    Fred Clare
#
#  Date of initial publication:
#    December, 2005
#
#  Description:
#    Create color wheels in HSV (hue, saturation, value) space.  
#    Each wheel is produced with a different value of V (the value 
#    parameter in HSV space).  Each wheel is composed of 16 wedges.  
#    The value for the hue remains the same within each wedge, and 
#    the value for the saturation varies linearly from the center 
#    (saturation = 0.) to the outer rim (saturation = 1.) within each 
#    wedge.  The hues vary from 0. to 360. counterclockwise around 
#    the color wheel starting at pure red (hue = 0.) and returning 
#    to pure red.
#
#    Also, this example contains basic tests for the six color
#    conversion functions:
#       Ngl.hsvrgb - (Hue, Saturation, Value) to (Red, Green, Blue)
#       Ngl.rgbhsv - (Red, Green, Blue) to (Hue, Saturation, Value)
#       Ngl.hlsrgb - (Hue, Lightness, Saturation) to (Red, Green, Blue)
#       Ngl.rgbhls - (Red, Green, Blue) to (Hue, Lightness, Saturation)
#       Ngl.yiqrgb - (luminance, chromaticity) to (Red, Green, Blue)
#       Ngl.rgbyiq - (Red, Green, Blue) to (luminance, chromaticity)
#
#  Effects illustrated:
#    o  Colors drawn in HSV space.
#    o  Converting from HSV space to RGB space.
#    o  Drawing lines and polygons in NDC space.
#    o  Conversions between the three color spaces: RGB, HSV, YIQ.
#
#  Output:
#    o Three color wheels are drawn using three different settings
#      for the "value" component in HSV space.
#    o Possible warnings to standard out if color conversion tests fail.
#

import Ngl 
import math
import numpy
#
#  Define the hues, and saturations, and values (the HSV) to be used.
#
hues        = Ngl.fspan(0., 337.5, 16)
saturations = Ngl.fspan(0., 1., 4)
values      = [0.50, 0.75, 1.00]

#
#  Define the radian increment to be used for spacing the
#  color wedges around the wheels.
#
pi = 3.14159265
radian_inc = 2*pi/float(len(hues))

#
#  Specify the Y-coordinates for the saturation labels.
#
slab_y = [.375, .292, .217]

#
#  Define Resource objects for the polygons and text.
#
poly_res = Ngl.Resources()
text_res = Ngl.Resources()
  
#
#  Set up arrays to hold the polygon coordinates.
#
x = numpy.zeros(4,'f')
y = numpy.zeros(4,'f')

#
#  Open a workstation with a black background and a white
#  foreground.  Specify a small color map so that there is 
#  enough room to add the 192 colors for all three color wheels.
#
rlist = Ngl.Resources()
rlist.wkForegroundColor = "White"
rlist.wkBackgroundColor = "Black"
rlist.wkColorMap        = "cyclic"
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"color3",rlist) 

#
#  Loop on the values, drawing a picture for each value.
#
for value in values:

#
#  Loop on the hues.
#
  for hue in hues:
     angle1 = (hue*len(hues)/360. - 0.5) * radian_inc
     angle2 = (hue*len(hues)/360. + 0.5) * radian_inc
     x[0] = x[3] = y[0] = y[3] = 0.0

#
#  Loop on the saturations.
#
     sindex = 0
     for saturation in saturations:
       r,g,b = Ngl.hsvrgb(hue,saturation,value)
       poly_res.gsFillColor = Ngl.new_color(wks,r,g,b)
       rlen = 0.25*(3.*saturation + 1.)
       x[1] = math.cos(angle1) * rlen
       y[1] = math.sin(angle1) * rlen
       x[2] = math.cos(angle2) * rlen
       y[2] = math.sin(angle2) * rlen
       xc = 0.1 + (x + 1.2)/3.  #  Conceptual user space is [-1.2, 1.2] in
       yc = 0.1 + (y + 1.2)/3.  #  X and Y -- map to [0.1, 0.9] NDC space.
       Ngl.polygon_ndc(wks,xc,yc,poly_res)
       x[0] = x[1]
       x[3] = x[2]
       y[0] = y[1]
       y[3] = y[2]
#
#  Label the saturation levels (the zero saturation 
#  level at the center of the wheel is not labeled).
#
       if (saturation != 0.):
         text_res.txFontHeightF = 0.022
         text_res.txJust = "CenterCenter" 
         slabel = "S = %4.2f" % (saturation)
         Ngl.text_ndc(wks, slabel, 0.5, slab_y[sindex],text_res)
         sindex = sindex+1

#
#  Add a main title specifying the "value".
#
  text_res.txFontHeightF = 0.03
  text_res.txJust = "CenterCenter" 
  Ngl.text_ndc(wks, "Value = %4.2f" % (value), 0.5, 0.9, text_res)

#
#  Mark the hues.
#
  text_res.txFontHeightF = 0.025
  text_res.txJust = "CenterLeft" 
  Ngl.text_ndc(wks, "Hue = 0.", 0.86, 0.5, text_res)
  Ngl.polyline_ndc(wks, [0.827, 0.843], [0.5, 0.5])
  Ngl.text_ndc(wks, "Hue = 45.", 0.767, 0.747, text_res)
  Ngl.polyline_ndc(wks, [0.730, 0.750], [0.730, 0.747])
  Ngl.text_ndc(wks, "Hue = 315.", 0.767, 0.253, text_res)
  Ngl.polyline_ndc(wks, [0.733, 0.752], [0.267, 0.251])
  text_res.txJust = "CenterRight" 
  Ngl.text_ndc(wks, "Hue = 135.", 0.233, 0.746, text_res)
  Ngl.polyline_ndc(wks, [0.270, 0.250], [0.731, 0.747])
  Ngl.text_ndc(wks, "Hue = 225.", 0.233, 0.254, text_res)
  Ngl.polyline_ndc(wks, [0.270, 0.250], [0.270, 0.253])

#
#  End picture.
#
  Ngl.frame(wks)

#
#  Illustrations and tests for the color conversion functions.
#

#
#  Set a tolerance limit for HLS and HSV tests.
#
eps = 0.00001

#
#  HLS to RGB: (120., 50., 100.) -> (1., 0., 0.)
#
r,g,b = Ngl.hlsrgb(120., 50., 100.)
if numpy.sometrue(numpy.greater(numpy.fabs([r-1., g, b]), eps)):
  print "color3: hlsrgb test failed"

#
#  RGB to HLS: (1., 0., 0.) -> (120., 50., 100.)
#
h,l,s = Ngl.rgbhls(1., 0., 0.)
if numpy.sometrue(numpy.greater(numpy.fabs([h-120.,l-50.,s-100.]), eps)):
 print "color3: rgbhls test failed"

#
#  HSV to RGB: (120., 1., 1.) -> (0., 1., 0.)
#
r,g,b = Ngl.hsvrgb(120., 1., 1.)   
if numpy.sometrue(numpy.greater(numpy.fabs([r, g-1., b]), eps)):
  print "color3: hsvrgb test failed"

#
#  RGB to HSV: (0., 1., 0.) -> (120., 1., 1.)
#
h,s,v = Ngl.rgbhsv(0., 1., 0.)        
if numpy.sometrue(numpy.greater(numpy.fabs([h-120., s-1., v-1.]), eps)):
 print "color3: hsvrgb test failed"

#
#  Set a tolerance limit for the YIQ tests.
#
eps = 0.01

#
#  YIQ to RGB: (0.58701, -0.27431, -0.52299) -> (0., 1., 0.)
#
r,g,b = Ngl.yiqrgb(0.58701, -0.27431, -0.52299)
if numpy.sometrue(numpy.greater(numpy.fabs([r, g-1., b]), eps)):
  print "color3: yiqrgb test failed"

#
#  RGB to YIQ: (1., 1., 1.) -> (1., 0., 0.)
#
y,i,q = Ngl.rgbyiq(1., 1., 1.)
if numpy.sometrue(numpy.greater(numpy.fabs([y-1., i, q]), eps)):
 print "color3: rgbyiq test failed"
      
Ngl.end()
