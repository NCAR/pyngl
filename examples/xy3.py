#
#  File:
#    xy3.py
#
#  Synopsis:
#    Illustrates how to draw stacked XY plots
#
#  Categories:
#    xy plots
#    polygons
#    polylines
#    text
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    January 2016
#
#  Description:
#    This example shows how to draw stacked XY plots with different
#    types of axes, and then highlights various parts of the plot
#    using polygons, lines, and text.
#
#  Effects illustrated:
#    o  Attaching primitives to a plot.
#    o  Attaching text to a plot.
#    o  Stacking plots.
# 
#  Output:
#     A single visualization with four plots is produced.
#
#  Notes:
#     
import numpy as np
import copy
import Ngl, os, sys 

#----------------------------------------------------------------------
# This function adds an annotation box to the given plot that 
# highlights a particular area with filled boxes and a title.
# "xmin"  and "xmax" represents which part of the x range to highlight.
#----------------------------------------------------------------------
def add_highlights(wks,plot,xmin,xmax,ymin,ymax,title):
  nboxes = 10
  xbox   = Ngl.fspan(xmin,xmax,nboxes)
  ybox   = [ymin,ymin,ymax,ymax,ymin]
  nboxes = xbox.shape[0]-1

#---Resources for filled purple boxes.
  gnres                 = Ngl.Resources()
  gnres.gsFillColor     = "goldenrod" # "MediumPurple1"
  gnres.gsFillOpacityF  = 0.15

  id = []
  for n in range(0,nboxes-1,2):
    id.append(Ngl.add_polygon(wks,plot,\
              [xbox[n],xbox[n+1],xbox[n+1],xbox[n],xbox[n]],\
              ybox,gnres))


#---Resources to outline box of interest
  lnres                  = Ngl.Resources()
  lnres.gsLineThicknessF = 3.0
  border = Ngl.add_polyline(wks,plot,[xmin,xmax,xmax,xmin,xmin],\
                                      [ymin,ymin,ymax,ymax,ymin],lnres)

  txres                 = Ngl.Resources()
  txres.txFontHeightF   = 0.022
  txres.txFont          = "Helvetica-Bold"
  txres.txJust          = "TopCenter"
  text = Ngl.add_text(wks,plot,title,(xmin+xmax)/2.,ymax-0.05,txres)

  return([id,border,text])

#----------------------------------------------------------------------
# Main code
#----------------------------------------------------------------------
#---Define the number of points in each curve.
NPTS  = 500
PI100 = 0.031415926535898

#---Create data for the four XY plots.
x     = np.array(range(NPTS))
theta = PI100*x
y1 = np.sin(theta)
y2 = np.sin(2**theta)
y3 = np.sin(4*np.sqrt(abs(theta)))
y4 = np.sin(theta * theta/7.)

#---Need for various axes
xmin  = x.min()
xmax  = x.max()
y1min = y1.min()
y1max = y1.max()
y2min = y2.min()
y2max = y2.max()
y3min = y3.min()
y3max = y3.max()
y4min = y4.min()
y4max = y4.max()

ymin  = y1min     # They should all be the same
ymax  = y1max

#---Open a PNG image for the graphics
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"xy3")

# Will be used to set shape and location of plots.
vph   = 0.20      # height
vpw   = 0.75      # width
vpx   = 0.12      # x location
vpy   = 0.95      # y location

#---Set resources common to all four plots
res                       = Ngl.Resources()
res.nglDraw               = False
res.nglFrame              = False
res.nglMaximize           = False

res.vpXF                  = vpx
res.vpWidthF              = vpw        # Make plots wider than 
res.vpHeightF             = vph        # they are high.

res.trXMinF               = xmin
res.trXMaxF               = xmax
res.trYMinF               = ymin-0.2    # Add a margin
res.trYMaxF               = ymax+0.2

res.xyLineThicknessF      = 5.         # 5x as thick

res.tiYAxisFont           = "Helvetica-Bold"
res.tiYAxisFontHeightF    = 0.02
res.tiYAxisFontThicknessF = 2.

res.tmYRMinorOn           = False
res.tmYLMinorOn           = False
res.nglPointTickmarksOutward = True

#---Set resources common to plots #1 and 3
res13                      = copy.copy(res)
res13.tmYLLabelFontHeightF = 0.015          # Increase font height
res13.tmYLLabelDeltaF      = 2.0            # Increase space b/w ticks and labels
res13.tmYLLabelJust        = "CenterLeft"   # left-justify labels
res13.tmYRLabelsOn         = False
res13.tmYROn               = False
            
#---Set resources common to plots #2 and 4
res24                      = copy.copy(res)
res24.tiYAxisSide          = "right"
res24.tiYAxisAngleF        = -90
res24.tmYUseLeft           = False          # Make right axis independent of left
res24.tmYLOn               = False          # Turn off left tickmarks
res24.tmYROn               = True           # Turn on right tickmarks
res24.tmYLLabelsOn         = False          # Turn off left labels
res24.tmYRLabelsOn         = True           # Turn on right labels
res24.tmYRLabelFontHeightF = 0.015          # Increase font height
res24.tmYRLabelDeltaF      = 2.5            # Increase space b/w ticks and labels
res24.tmYRLabelJust        = "CenterRight"  # right-justify labels

#---Copy over common resources to the individual plot resources.
res1 = copy.copy(res13)
res2 = copy.copy(res24)
res3 = copy.copy(res13)
res4 = copy.copy(res24)

#---Plot #1
res1.vpYF              = vpy
res1.xyLineColor       = "Seagreen"
res1.tiYAxisFontColor  = res1.xyLineColor 
res1.tiYAxisString     = "xy1"
res1.tmXBBorderOn      = False 
res1.tmXBOn            = False 

xy1 = Ngl.xy(wks,x,y1,res1)

#---Plot #2
res2.vpYF              = res1.vpYF - vph
res2.xyLineColor       = "Brown"
res2.tiYAxisString     = "xy2"
res2.tiYAxisFontColor  = res2.xyLineColor 
res2.tmXTOn            = False 
res2.tmXBOn            = False 
res2.tmXTBorderOn      = False 
res2.tmXBBorderOn      = False 

xy2 = Ngl.xy(wks,x,y2,res2)

#---Plot #3
res3.vpYF              = res2.vpYF - vph
res3.xyLineColor       = "NavyBlue"
res3.tiYAxisString     = "xy3"
res3.tiYAxisFontColor  = res3.xyLineColor 
res3.tmXTOn            = False 
res3.tmXTBorderOn      = False 
res3.tmXBOn            = False 
res3.tmXBBorderOn      = False 

xy3 = Ngl.xy(wks,x,y3,res3)

#---Plot #4
res4.vpYF              = res3.vpYF - vph
res4.xyLineColor       = "Magenta4"
res4.tiYAxisString     = "xy4"
res4.tiYAxisFontColor  = res4.xyLineColor 
res4.tmXTBorderOn      = False 
res4.tmXTOn            = False 

xy4 = Ngl.xy(wks,x,y4,res4)

#---Add special highlighting to the given plots
add_highlights(wks,xy1, 50,250,y1min-0.2,y1max+0.2,"highlight #1")
add_highlights(wks,xy4,300,500,y4min-0.2,y4max+0.2,"highlight #4")

#---Draw all four plots and advance the frame.
Ngl.draw(xy1)
Ngl.draw(xy2)
Ngl.draw(xy3)
Ngl.draw(xy4)
Ngl.frame(wks)

Ngl.end()

