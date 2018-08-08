#
#  File:
#    contour2.py
#
#  Synopsis:
#    Illustrates how to fill a contour plot with colors and patterns.
#
#  Categories:
#    Contouring
#
#  Author:
#    Fred Clare
#  
#  Date of initial publication:
#    February 2004
#
#  Description:
#    This example shows how to fill a contour with different patterns
#    and colors.
#
#  Effects illustrated:
#    o  Using resources to change fill patterns and colors.
#    o  Changing the color map.
#    o  Using indexed color.
# 
#  Output:
#     A single visualization is produced.
#
#  Notes:
#     
from __future__ import print_function
import numpy
import Ngl

#
# Create some data for the contour plot.
#
N=25

T = numpy.zeros((N,N),'f')

jspn = numpy.power(numpy.arange(-12,13),2)
ispn = numpy.power(numpy.arange(-12,13),2)

for i in range(0,len(ispn)):
  T[i,:] = (jspn + ispn[i]).astype('f')

T = 100.0 - numpy.sqrt(64 * T)

#
# Open the workstation and change the color map.
#
rlist    = Ngl.Resources()
rlist.wkColorMap = "default"
wks_type = "png"

wks = Ngl.open_wks(wks_type,"contour2",rlist)

cnres                   = Ngl.Resources()
cnres.cnFillOn          = True   # Turn on contour level fill.
cnres.cnMonoFillPattern = False  # Indicate you want multiple fill patterns.

#
# Set cnFillPatterns and cnFillColors to various indexes representing
# fill patterns and colors.  A fill pattern index of "0" is solid fill.
# If you don't want any fill for a particular contour level, set it
# to "-1," which means "transparent."
#
cnres.cnFillPatterns    = [0, 0, 0, 2, 3, 6, 8,10,-1, 9, 11,12,17,16]
cnres.cnMonoFillScale   = False    # We want to use multiple fill scales.
cnres.cnFillScales      = [1.,.5,.6,1.,.7,.5,.4,.9,.6,1.,1.,.8,.5,.9]
cnres.cnFillColors      = [ 6, 8,20,22, 4, 2,10, 1,12,15, 1,13, 5, 7]
  
contour = Ngl.contour(wks,T,cnres)
Ngl.end()
