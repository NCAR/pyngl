#
#  File:
#    ndc_grid.py
#
#  Synopsis:
#    Illustrates how to draw primitives and text using NDC coordinates.
#
#  Categories:
#    polygons
#    polylines
#    polymarkers
#    text
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    April 2008
#
#  Description:
#    This example shows how to draw primitives and text using
#    NDC coordinates. The draw_ndc_grid function is used as
#    a tool to help determine which coordinates to use.

#  Effects illustrated:
#    o  Drawing NDC primitives.
#    o  Setting resources for primitives.
#    o  Using the NDC grid routine for debugging purposes.
#    o  Drawing primitives with missing values
#    o  Using indexed color.
# 
#  Output:
#     A single visualization with an NDC grid and primitives is produced.
#
#  Notes:
#     

#
#  Import NumPy, Ngl functions
#
import numpy, Ngl 

# Import masked arrays
from numpy import ma

#
# Draw some curves, with and without missing values, and include
# some markers and text as references.
#
def draw_lines():
  mkres = Ngl.Resources()    # Marker resources
  lnres = Ngl.Resources()    # Line resources

  xline  = Ngl.fspan(0.15,0.85,80)
  yline1 = 0.025 * (numpy.cos(2.*PI*(Ngl.fspan(1,80,80)/20.))) + 0.9
  yline2 = 0.045 * (numpy.sin(2.*PI*(Ngl.fspan(1,80,80)/20.))) + 0.8
  yline3 = 0.045 * (numpy.sin(2.*PI*(Ngl.fspan(1,80,80)/10.))) + 0.7

# Draw 3 sets of lines.
  lnres.gsLineColor      = "forestgreen"      # Black is the default
  lnres.gsLineThicknessF = 4.0                # 1.0 is the default
  Ngl.polyline_ndc(wks,xline,yline1,lnres)

  lnres.gsLineColor       = "brown4"
  lnres.gsLineThicknessF  = 3.0
  lnres.gsLineDashPattern = 2     # Default is a solid line (0)
  Ngl.polyline_ndc(wks,xline,yline2,lnres)

  lnres.gsLineColor       = "NavyBlue"
  lnres.gsLineThicknessF  = 2.0
  lnres.gsLineDashPattern = 11
  Ngl.polyline_ndc(wks,xline,yline3,lnres)

# Draw markers at the end of each of the 3 lines.
  mkres.gsMarkerColor = "ForestGreen"
  mkres.gsMarkerIndex = 6
  mkres.gsMarkerSizeF = 0.02
  Ngl.polymarker_ndc(wks,[xline[0],xline[-1]],[yline1[0],yline1[-1]],mkres)

  mkres.gsMarkerColor = "brown4"
  mkres.gsMarkerIndex = 11
  Ngl.polymarker_ndc(wks,[xline[0],xline[-1]],[yline2[0],yline2[-1]],mkres)

  mkres.gsMarkerColor = "NavyBlue"
  mkres.gsMarkerIndex = 16
  mkres.gsMarkerSizeF = 0.01
  Ngl.polymarker_ndc(wks,[xline[0],xline[-1]],[yline3[0],yline3[-1]],mkres)

#
# Draw a vertical line with some missing values. Include markers
# so you can see where the missing values were.
#
  lnres.gsLineColor       = "blue4"
  lnres.gsLineThicknessF  = 1.5
  lnres.gsLineDashPattern = 0
  vxline = ma.array(yline2[25::3] - .65)
  vyline = ma.array( xline[25::3] - .30)
#
# Draw the markers for each point.
#
  mkres.gsMarkerColor      = "mediumorchid4"
  mkres.gsMarkerIndex      = 12     # Stars
  mkres.gsMarkerThicknessF = 2.0    # Default is 1.0
  Ngl.polymarker_ndc(wks,vxline,vyline,mkres)
#
# Make vxline a masked array and set some values to missing
# using the "mask" array.  Note that if there's a single point
# surrounded by missing values, it will become a marker when
# drawn with Ngl.polyline_ndc.
#
  vxline = ma.array(vxline,mask=[0,1,0,1,0,0,0,0,0,1,1,0,0,1,0,0,0,1,0])
  Ngl.polyline_ndc(wks,vxline,vyline,lnres)
#
# Draw a text string describing the above line.
#
  tres               = Ngl.Resources()
  tres.txAngleF      = -90.
  tres.txFontHeightF = 0.03
  tres.txJust        = "CenterCenter"
  Ngl.text_ndc(wks,"Line with msg vals",0.23,0.31,tres)

#
# Put a box around the missing value curve.
#
  bres                  = Ngl.Resources()
  bres.gsLineColor      = "lightsalmon4"
  bres.gsLineThicknessF = 3.0
  Ngl.polyline_ndc(wks,[.09,.28,.28,.09,.09],[.04,.04,.57,.57,.04],bres)
	
#
# Draw some polygons, filled and shaded.
#
def draw_gons(xshift=0.0,yshift=0.0):
  gnres = Ngl.Resources()    # Polygon resources

# The filled diamond
  gnres.gsFillColor = "lightpink1"
  xdia = numpy.array([0.00,0.10,0.20,0.10,0.00]) + xshift
  ydia = numpy.array([0.10,0.00,0.10,0.20,0.10]) + yshift
  Ngl.polygon_ndc(wks,xdia,ydia,gnres)

# The filled four squares
  gnres.gsFillColor = "lightslateblue"
  xbox = numpy.array([0.00,0.05,0.05,0.00,0.00]) + xshift
  ybox = numpy.array([0.00,0.00,0.05,0.05,0.00]) + yshift

  Ngl.polygon_ndc(wks,xbox,     ybox,     gnres)
  Ngl.polygon_ndc(wks,xbox+0.15,ybox,     gnres)
  Ngl.polygon_ndc(wks,xbox,     ybox+0.15,gnres)
  Ngl.polygon_ndc(wks,xbox+0.15,ybox+0.15,gnres)
  Ngl.polyline_ndc(wks,xbox,     ybox)
  Ngl.polyline_ndc(wks,xbox+0.15,ybox)
  Ngl.polyline_ndc(wks,xbox,     ybox+0.15)
  Ngl.polyline_ndc(wks,xbox+0.15,ybox+0.15)

# The shaded slanted boxes attaching the squares
  gnres.gsFillColor = "cyan"
  xbt1 = numpy.array([0.05,0.10,0.10,0.05,0.05]) + xshift
  ybt1 = numpy.array([0.00,0.05,0.10,0.05,0.00]) + yshift
  xbt2 = ybt1
  ybt2 = xbt1
  xbt3 = xbt1 + .05
  ybt3 = ybt1 + .10
  xbt4 = ybt3
  ybt4 = xbt3
  gnres.gsFillColor = "black"
  gnres.gsFillIndex = 17
  Ngl.polygon_ndc(wks,xbt1,ybt1,gnres)
  Ngl.polygon_ndc(wks,xbt2,ybt2,gnres)
  Ngl.polygon_ndc(wks,xbt3,ybt3,gnres)
  Ngl.polygon_ndc(wks,xbt4,ybt4,gnres)
  Ngl.polyline_ndc(wks,xbt1,ybt1,gnres)
  Ngl.polyline_ndc(wks,xbt2,ybt2,gnres)
  Ngl.polyline_ndc(wks,xbt3,ybt3,gnres)
  Ngl.polyline_ndc(wks,xbt4,ybt4,gnres)

  xtb1 = numpy.array([0.10,0.15,0.15,0.10,0.10]) + xshift
  ytb1 = numpy.array([0.05,0.00,0.05,0.10,0.05]) + yshift
  xtb2 = ytb1
  ytb2 = xtb1
  xtb3 = xtb1 - .05
  ytb3 = ytb1 + .10
  xtb4 = ytb3
  ytb4 = xtb3
  Ngl.polygon_ndc(wks,xtb1,ytb1,gnres)
  Ngl.polygon_ndc(wks,xtb2,ytb2,gnres)
  Ngl.polygon_ndc(wks,xtb3,ytb3,gnres)
  Ngl.polygon_ndc(wks,xtb4,ytb4,gnres)
  Ngl.polyline_ndc(wks,xtb1,ytb1,gnres)
  Ngl.polyline_ndc(wks,xtb2,ytb2,gnres)
  Ngl.polyline_ndc(wks,xtb3,ytb3,gnres)
  Ngl.polyline_ndc(wks,xtb4,ytb4,gnres)

#
# Draw some text showing how to change the justication
# and direction.
#
def draw_text():
  txres = Ngl.Resources()    # Text resources
  mkres = Ngl.Resources()    # Marker resources

  txres.txFontHeightF = 0.03
  txres.txFontColor   = "black"
  txres.txPerimOn     = True
  txres.txBackgroundFillColor = "yellow"

  mkres.gsMarkerIndex = 5      # "X"
  mkres.gsMarkerSizeF = 0.01
  mkres.gsMarkerThicknessF = 2.5
  mkres.gsMarkerColor = "red"

  txres.txJust        = "BottomCenter"
  Ngl.text_ndc(wks,"_"+txres.txJust+"_",0.6,0.3,txres)
  Ngl.polymarker_ndc(wks,0.6,0.3,mkres)

  txres.txJust        = "TopLeft"
  Ngl.text_ndc(wks,"_"+txres.txJust+"_",0.6,0.25,txres)
  Ngl.polymarker_ndc(wks,0.6,0.25,mkres)

  txres.txJust        = "CenterRight"
  Ngl.text_ndc(wks,"_"+txres.txJust+"_",0.6,0.15,txres)
  Ngl.polymarker_ndc(wks,0.6,0.15,mkres)

  txres.txJust        = "TopCenter"
  txres.txDirection   = "Down"
  Ngl.text_ndc(wks,txres.txJust,0.85,0.5,txres)
  Ngl.polymarker_ndc(wks,0.85,0.5,mkres)

#
# Main code
#
PI = 3.14159

#
# Send graphics PNG file.
#
wks_type = "png"
wks = Ngl.open_wks(wks_type,"ndc_grid")

# Comment this out to get rid of the gray NDC grid.
Ngl.draw_ndc_grid(wks)

draw_lines()
draw_gons(xshift=0.4,yshift=0.4)
draw_text()

Ngl.frame(wks)

Ngl.end()
