#
#  File:
#    xy1.py
#
#  Synopsis:
#    Illustrates how to change the X and Y axes from linear to log.
#
#  Categories:
#    xy plots
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    January 2006
#
#  Description:
#    This example shows how to generate four XY plots, each with a
#    different style of axes.
#
#  Effects illustrated:
#    o  Using transformation resources to change from linear to log.
#    o  Panelling XY plots using viewport resources.
#    o  Setting tickmark resources to get different sets and styles 
#       of axes drawn.
# 
#  Output:
#     A single visualization with four plots is produced.
#
#  Notes:
#     

import Ngl
import math
import numpy

#
# Create dummy data for XY plot.
#

npts  = 501
xspan = Ngl.fspan(0,npts-1,npts)
x     = numpy.zeros([npts],'f')
y     = numpy.zeros([npts],'f')

for i in range(0,npts):
  x[i] = 500.+.9*xspan[i]*math.cos(0.031415926535898*xspan[i])
  y[i] = 500.+.9*xspan[i]*math.sin(0.031415926535898*xspan[i])

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"xy1")

#
# Even though some resources are the same, create four different
# resource lists for each XY plot we plan to create.
#
xyres1 = Ngl.Resources()
xyres2 = Ngl.Resources()
xyres3 = Ngl.Resources()
xyres4 = Ngl.Resources()

xyres1.nglMaximize        = False   # Turn this off, let vp* resources work.
xyres1.nglFrame           = False   # Don't advance frame yet.

xyres1.vpXF               = 0.1     # Change X,Y position of plot.
xyres1.vpYF               = 0.93
xyres1.vpHeightF          = 0.32
xyres1.vpWidthF           = 0.32

xyres1.tiMainString       = "Perimeter Background"  # Set some titles
xyres1.tiXAxisString      = "Linear"
xyres1.tiYAxisString      = "Linear"

xyres1.trXMaxF            = 900.      # Set limits for X,Y axes
xyres1.trXMinF            = 0.
xyres1.trYMaxF            = 1000.
xyres1.trYMinF            = 100.

xyres1.xyLineColor        = "red"
xyres1.xyLineThicknessF   = 3.0

xyres2.nglMaximize        = False   # Turn this off, let vp* resources work.
xyres2.nglFrame           = False   # Don't advance frame yet.

xyres2.vpXF               = 0.6     # Change X,Y position of plot.
xyres2.vpYF               = 0.93
xyres2.vpHeightF          = 0.32
xyres2.vpWidthF           = 0.32

xyres2.tiMainString       = "Grid Background"  # Set some titles
xyres2.tiXAxisString      = "Linear" 
xyres2.tiYAxisString      = "Logarithmic"

xyres2.tmXMajorGrid       = True      # Turn on X grid lines

xyres2.trXMaxF            = 900.      # Set limits for X,Y axes
xyres2.trXMinF            = 0.
xyres2.trYMaxF            = 1000.
xyres2.trYMinF            = 100.

xyres2.xyLineColor        = "blue"
xyres2.xyLineThicknessF   = 3.0

xyres2.xyYStyle           = "Log"   # Convert Y axis to log.

xyres3.nglMaximize        = False   # Turn this off, let vp* resources work.
xyres3.nglFrame           = False   # Don't advance frame yet.

xyres3.vpXF               = 0.1     # Change X,Y position of plot.
xyres3.vpYF               = 0.45
xyres3.vpHeightF          = 0.32
xyres3.vpWidthF           = 0.32

xyres3.tiMainString       = "Half-Axis Background"  # Set some titles
xyres3.tiXAxisString      = "Logarithmic"
xyres3.tiYAxisString      = "Linear" 

xyres3.tmXTBorderOn       = False      # Change some tickmark resources.
xyres3.tmXTOn             = False
xyres3.tmYRBorderOn       = False
xyres3.tmYROn             = False

xyres3.trXMaxF            = 1000.      # Set limits for X,Y axes
xyres3.trXMinF            = 10.
xyres3.trYMaxF            = 1000.
xyres3.trYMinF            = 100.

xyres3.xyLineColor        = "green"
xyres3.xyLineThicknessF   = 3.0

xyres3.xyXStyle           = "Log"   # Convert X axis to log.

xyres4.nglMaximize        = False   # Turn this off, let vp* resources work.

xyres4.vpXF               = 0.6     # Change X,Y position of plot.
xyres4.vpYF               = 0.45
xyres4.vpHeightF          = 0.32
xyres4.vpWidthF           = 0.32

xyres4.tiMainString       = "No Background"  # Set some titles
xyres4.tiXAxisString      = "Logarithmic"
xyres4.tiYAxisString      = "Logarithmic"

xyres4.tmXBBorderOn       = False      # Change some tickmark resources.
xyres4.tmXBOn             = False
xyres4.tmXTBorderOn       = False
xyres4.tmXTOn             = False
xyres4.tmYLBorderOn       = False
xyres4.tmYLOn             = False
xyres4.tmYRBorderOn       = False
xyres4.tmYROn             = False

xyres4.trXMaxF            = 1000.      # Set limits for X,Y axes
xyres4.trXMinF            = 31.628
xyres4.trYMaxF            = 1000.
xyres4.trYMinF            = 100.

xyres4.xyLineColor        = "orange"
xyres4.xyLineThicknessF   = 3.0

xyres4.xyXStyle           = "Log"   # Convert X axis to log.
xyres4.xyYStyle           = "Log"   # Convert Y axis to log.

xy1 = Ngl.xy(wks,x,y,xyres1)
xy2 = Ngl.xy(wks,x,y,xyres2)
xy3 = Ngl.xy(wks,x,y,xyres3)
xy4 = Ngl.xy(wks,x,y,xyres4)


Ngl.end()
