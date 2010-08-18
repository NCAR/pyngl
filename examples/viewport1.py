#
#  File:
#    viewport1.py
#
#  Synopsis:
#    Illustrates the difference between the viewport and bounding box.
#
#  Categories:
#    viewport
#    polylines
#    polymarkers
#    text
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    August 2010
#
#  Description:
#    This example shows how to raw primitives and text using
#    NDC coordinates. The draw_ndc_grid function is used as
#    a tool to help determine which coordinates to use.

#  Effects illustrated:
#    o  Drawing a simple filled contour plot
#    o  Drawing a box around a contour plot viewport
#    o  Drawing the bounding box
#    o  Changing the color and thickness of polylines
#    o  Drawing polylines, polymarkers, and text in NDC space
#    o  Adding named colors to an existing color map
#    o  Using "getvalues" to retrieve resource values
#    o  Generating dummy data
# 
#  Output:
#     A single visualization with the viewport and bounding box 
#     information included.
#
#  Notes:
#     

import numpy, Ngl

#********************************************************************
# Draw a box around the viewport of the given object..
#********************************************************************
def draw_vp_box(wks,plot):

# Retrieve the viewport values of the drawable object.
  vpx = Ngl.get_float(plot,"vpXF")
  vpy = Ngl.get_float(plot,"vpYF")
  vpw = Ngl.get_float(plot,"vpWidthF")
  vph = Ngl.get_float(plot,"vpHeightF")

  print "Viewport x,y,width,height =",vpx,vpy,vpw,vph

# Make a box with the viewport values.
  xbox = [vpx,vpx+vpw,vpx+vpw,vpx,vpx]
  ybox = [vpy,vpy,vpy-vph,vpy-vph,vpy]

# Set up some marker resources.
  mkres                  = Ngl.Resources()
  mkres.gsMarkerIndex    = 16     # filled dot
  mkres.gsMarkerSizeF    = 0.02   # larger than default
  mkres.gsMarkerColor    = "ForestGreen"

# Draw a single marker at the vpXF/vpYF location.
  Ngl.polymarker_ndc(wks,vpx,vpy,mkres)

# Set up some line resources.
  lnres                  = Ngl.Resources()
  lnres.gsLineColor      = "NavyBlue"  # line color
  lnres.gsLineThicknessF = 3.5         # 3.5 times as thick

# Draw a box around the viewport.
  Ngl.polyline_ndc(wks,xbox,ybox,lnres)

# Set up some text resources.
  txres                       = Ngl.Resources()  
  txres.txJust                = "CenterLeft"
  txres.txFontHeightF         = 0.015
  txres.txFontColor           = "ForestGreen"
  txres.txBackgroundFillColor = "white"

# Draw a text string labeling the marker
  Ngl.text_ndc(wks,"(vpXF,vpYF)",vpx+0.03,vpy,txres)

# Draw text strings labeling the viewport box.
  txres.txFontColor           = "black"
  txres.txJust                = "CenterLeft"

  Ngl.text_ndc(wks,"viewport",vpx+vpw/2.,vpy-vph,txres)
  Ngl.text_ndc(wks,"viewport",vpx+vpw/2.,vpy,txres)

  txres.txAngleF = 90.
  Ngl.text_ndc(wks,"viewport",vpx,vpy-vph/2.,txres)
  Ngl.text_ndc(wks,"viewport",vpx+vpw,vpy-vph/2.,txres)

  return

#********************************************************************
# Draw a box around the bounding box of the given object..
#********************************************************************
def draw_bb_box(wks,plot):
# Retrieve the bounding box of the given object.
  bb  = Ngl.get_bounding_box(plot)
  top = bb[0]
  bot = bb[1]
  lft = bb[2]
  rgt = bb[3]

  print "Bounding box top,bottom,left,right =",top,bot,lft,rgt

# Make a box with the bounding box values.
  xbox = [rgt,lft,lft,rgt,rgt]
  ybox = [top,top,bot,bot,top]

# Set up some line resources.
  lnres                   = Ngl.Resources()
  lnres.gsLineColor       = "Brown"
  lnres.gsLineThicknessF  = 2.5

# Set up some text resources.
  txres                       = Ngl.Resources()
  txres.txFontHeightF         = 0.015
  txres.txBackgroundFillColor = "white"
  txres.txJust                = "CenterLeft"

# Draw a box showing the bounding box.
  Ngl.polyline_ndc(wks,xbox,ybox,lnres)

# Draw text strings labeling the bounding box.

  Ngl.text_ndc(wks,"bounding box",lft+0.05,bot,txres)
  txres.txJust                = "CenterRight"
  Ngl.text_ndc(wks,"bounding box",rgt-0.05,top,txres)

  txres.txAngleF              = 90.
  txres.txJust                = "CenterRight"
  Ngl.text_ndc(wks,"bounding box",lft,top-0.05,txres)

  txres.txJust                = "CenterLeft"
  Ngl.text_ndc(wks,"bounding box",rgt,bot+0.05,txres)

  return

#********************************************************************
# Main code
#********************************************************************
wks_type = "ps"
rlist               = Ngl.Resources()
rlist.wkColorMap    = "nrl_sirkes"
rlist.wkOrientation = "Portrait"             # Don't let it be landscape
wks = Ngl.open_wks("ps","viewport1",rlist)   # Open a PS workstation.

# Add some named colors.
forest_green = numpy.array([ 34, 139,  34])/255.
navy_blue    = numpy.array([  0,   0, 128])/255.
brown        = numpy.array([165,  42,  42])/255.
ig = Ngl.new_color(wks,forest_green[0],forest_green[1],forest_green[2])
ib = Ngl.new_color(wks,navy_blue[0],   navy_blue[1],   navy_blue[2])
ir = Ngl.new_color(wks,brown[0],       brown[1],       brown[2])

# Generate some dummy data.

cmin = -19.23
cmax =  16.81

data = Ngl.generate_2d_array([100,100], 10, 10, cmin, cmax)
nice_min,nice_max,nice_spc = Ngl.nice_cntr_levels(cmin,cmax,cint=3)

# Set up resources for a contour plot.

cnres                   = Ngl.Resources()

cnres.nglMaximize       = True      # Maximize plot in frame
cnres.nglDraw           = False     # Don't draw plot
cnres.nglFrame          = False     # Don't advance the frame

cnres.cnFillOn          = True        # Turn on contour fill
cnres.nglSpreadColorEnd = -4          # Skip last three colors

cnres.cnLevelSelectionMode = "ManualLevels"
cnres.cnLevelSpacingF      = nice_spc
cnres.cnMinLevelValF       = nice_min
cnres.cnMaxLevelValF       = nice_max

cnres.lbOrientation     = "Vertical"  # Default is horizontal

cnres.tiMainString      = "This is a title"
cnres.tiXAxisString     = "X axis"
cnres.tiYAxisString     = "Y axis"

contourplot = Ngl.contour(wks,data,cnres)

# Draw plot with viewport and bounding boxes.
Ngl.draw(contourplot)
draw_bb_box(wks,contourplot)
draw_vp_box(wks,contourplot)

# Advance frame.
Ngl.frame(wks)

Ngl.end()

