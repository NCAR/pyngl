#
#  File:
#    datatondc01.py
#
#  Synopsis:
#    Illustrates converting data coordinates to NDC coordinates.
#
#  Category:
#    Processing.  
#
#  Author:
#    Fred Clare
#
#  Date of initial publication:
#    September, 2007
#
#  Description:
#    Draws an xy plot and then places markers along the curve drawn.
#
#  Effects illustrated:
#    o  Converting data coordinates to NDC coordinates.
#    o  Drawing an xy plot.
#    o  Drawing polymarkers.
#    o  Converting from data to NDC and back to data.
#
#  Output:
#    Two visualizations are produced:
#      1.)  A simple xy plot of a sine curve.
#      2.)  The same plot as in 1.), but with polymarkers drawn on it.
#
#  Notes:
#
import Ngl
import numpy

#
#  Create some data for xy plots.
#
npts = 400
x = Ngl.fspan(100.,npts-1,npts)
y = 500.+ x * numpy.sin(0.031415926535898*x)

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"datatondc1")

res = Ngl.Resources()
res.nglMaximize = False 
res.vpXF        = 0.1  
res.vpYF        = 0.9
res.vpHeightF   = 0.8
res.vpWidthF    = 0.8

xy = Ngl.xy(wks,x,y,res)

x_in = x
y_in = numpy.absolute(y)

x_out, y_out = Ngl.datatondc(xy,x_in,y_in)

#
#  Print out a table of data versus NDC values.
#
# for i in xrange(len(x_out)):
#   print "%4d  data: (%3.0f, %8.4f)  NDC: (%6.4f, %6.4f)" % \
#                  (i,x_in[i],y_in[i],x_out[i],y_out[i])

#
#  Draw some markers on the curve in the above plot.
#
Ngl.draw(xy)   #  Redraw the plot to start with.

#
#  Then draw markers.
#
gsres = Ngl.Resources()
gsres.gsMarkerColor = "Blue"
gsres.gsMarkerIndex = 16
gsres.gsMarkerSizeF = 17.0
Ngl.polymarker_ndc(wks,x_out[::8],y_out[::8],gsres)

gsres.gsMarkerColor = "Red"
gsres.gsMarkerIndex = 12
gsres.gsMarkerSizeF = 22.0
Ngl.polymarker_ndc(wks,x_out[4::8],y_out[4::8],gsres)

Ngl.frame(wks)

#
#  Show that datatondc and ndctodata are inverse functions by
#  converting a data point to NDC and then converting that point
#  back to data space.
#
x_ndc, y_ndc = Ngl.datatondc(xy, 100., 600.)
x_dat, y_dat = Ngl.ndctodata(xy, x_ndc, y_ndc)
#
#  Optional print.
#
#  print "Original data:    (100.0000, 600.0000)  \nNDC:              (%06.4f, %06.4f)  \nBack to original: (%6.4f, %6.4f)" % (x_ndc,y_ndc,x_dat,y_dat)

Ngl.end()
