#
#  File:
#    fillxy2.py
#
#  Synopsis:
#    Illustrates how to fill between curves in an XY plot.
#
#  Categories:
#    xy plots
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    August 2010, based on NCL example
#
#  Description:
#    This example illustrates the creation of multiple curves
#    curves, with various filling between them.
#
#  Effects illustrated:
#    o  Filling the area between multiple curves in an XY plot
#    o  Pointing tickmarks outward
# 
#  Output:
#     Two visualizations with different types of fill.
#

import numpy,Ngl


#
# Define the number of points in each curve.
#
NPTS  = 500
PI100 = 0.031415926535898
#
# Create data for the four XY plots.
#
y      = numpy.zeros([4,NPTS],'f')
x      = numpy.arange(0,NPTS,1)
theta  = PI100*x
y[0,:] = numpy.sin(theta)
y[1,:] = 2+numpy.sin(2*numpy.sqrt(numpy.fabs(theta)))   # Make sure they
y[2,:] = 4+numpy.sin(3*numpy.sqrt(numpy.fabs(theta)))   # don't intersect.
y[3,:] = 6+numpy.sin(10*numpy.sqrt(numpy.fabs(theta)))

rlist = Ngl.Resources()
rlist.wkColorMap = ["white","black","darkgreen","dodgerblue4",\
                    "orange","blue","coral","brown","purple","red"]
rlist.wkOrientation = "Portrait"

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"fillxy2",rlist)

#
# Specify the colors to use between adjacent curves.
#
# The area b/w curves y(0,:) and y(1,:) will be filled in with red
# The area b/w curves y(1,:) and y(2,:) will be filled in with blue
# The area b/w curves y(2,:) and y(3,:) will be filled in with orange
#
res                  = Ngl.Resources()     # Plot options desired

res.tiMainString     = "nglXYFillColors = 'red','blue','orange'"

res.xyLineColor      = -1     # Don't draw the line itself

res.nglXYFillColors  = ["red","blue","orange"]

res.xyLabelMode      = "Custom"           # custom label the curves
res.xyExplicitLabels = ['y1','y2','y3','y4']

res.nglPointTickmarksOutward = True   # Point tickmarks outward.

xy = Ngl.xy(wks,x,y,res)    # Draw the four curves with fill

#
# The second plot will fill curves differently, depending on
# where they intersect.
#

# Create 6 curves.
ncurves = 6
y = numpy.zeros([ncurves,NPTS],'f')
for i in range(0,ncurves):
  y[i,:] = numpy.sin((i+1)*numpy.sqrt(numpy.fabs(theta)))


# Space the curves out a little.
y[2,:] = 2 + y[2,:]
y[3,:] = 2 + y[3,:]
y[4,:] = 4 + y[4,:]
y[5,:] = 4 + y[5,:]

#
# Specify the colors to use between adjacent curves, depending
# on where they intersect.
#
# "purple" will be used to fill all areas where curve y[1,:] > y[0,:]
# "orange" will be used to fill all areas where curve y[1,:] < y[0,:]
#
# "brown" will be used to fill all areas where curve y[3,:] > y[2,:]
# "coral" will be used to fill all areas where curve y[3,:] < y[2,:]
#
# "darkgreen" will be used to fill all areas where curve y[5,:] > y[4,:]
# "dodgerblue4" will be used to fill all areas where curve y[5,:] < y[4,:]
#
# Nothing will be done between curves y[1,:] & y[2,:] or
#     curves y[3,:] & y[4,:]  (hence the "transparent" setting)
#

#---Whether to fill curves above/below or right/left.
DO_RIGHT_LEFT = True

res = Ngl.Resources()

colors1 = ["purple","transparent","brown","transparent","darkgreen"]
colors2 = ["orange","transparent","coral","transparent","dodgerblue4"]

res.nglPointTickmarksOutward = True   # Point tickmarks outward.
res.xyLabelMode              = "Custom"           # custom label the curves
res.xyExplicitLabels         = ['y1','y2','y3','y4','y5','y6']

if DO_RIGHT_LEFT:
  res.tiMainString         = "nglXYRightFillColors/nglXYLeftFillColors"
  res.nglXYRightFillColors = colors1
  res.nglXYLeftFillColors  = colors2

  xy = Ngl.xy(wks,y,x,res)    # Draw the six curves with fill
else:
  res.tiMainString         = "nglXYAboveFillColors/nglXYBelowFillColors"
  res.nglXYAboveFillColors = colors1
  res.nglXYBelowFillColors = colors2

  xy = Ngl.xy(wks,x,y,res)    # Draw the six curves with fill

Ngl.end()
