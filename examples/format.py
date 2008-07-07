#
#  File:
#    format.py
#
#  Synopsis:
#    Illustrates how to format the tickmark labels.
#
#  Categories:
#    xy plots
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    July 2008
#
#  Description:
#    This example shows several methods for formatting tickmark labels.
#    This formatting method can be used for other resources, like
#    "cnLineLabelFormat".
#
#  Effects illustrated:
#    o  Drawing multiple tickmark objects on a frame (page), using
#    o  ViewPort (vp) resources to reposition each object.
#    o  Setting format resources to control the tickmark labels.
#    o  Drawing text strings in NDC space.
#    o  Using Ngl.set_values to change resources.
# 
#  Output:
#     A single visualization with several tickmark objects is produced.
#
#  Notes:
#     

import Ngl
import numpy

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"format")

#
# Create some dummy data for an XY plot. We are only doing this
# to create an object that has tickmarks.
#
x = Ngl.fspan(0,1,20)
y = x

#
# Set some resouces for a dummy XY plot.
#
res = Ngl.Resources()
res.nglMaximize          = False
res.nglDraw              = False
res.nglFrame             = False

res.xyLineColor          = -1       # Transparent line

#
# Resources for main title and Y axis title.
#
res.tiMainFont           = "Helvetica-Bold"
res.tiMainFontHeightF    = 0.015
res.tiMainJust           = "CenterLeft"
res.tiMainPosition       = "Left"
res.tiYAxisAngleF        = 0.0
res.tiYAxisFont          = "Courier-Bold"
res.tiYAxisFontAspectF   = 1.5
res.tiYAxisFontHeightF   = 0.015
res.tiYAxisFuncCode      = "|"
res.tiYAxisJust          = "BottomRight"
	
#
# Resources for X and Y axes. The top, left, and right axes
# will be turned off.
#
res.tmXTOn               = False      # Turn off top tickmarks
res.tmYLOn               = False      # Turn off left tickmarks
res.tmYROn               = False      # Turn off bottom tickmarks
res.tmXTBorderOn         = False      # Turn off top border
res.tmYLBorderOn         = False      # Turn off left border
res.tmYRBorderOn         = False      # Turn off right border

#
# Resources for the tickmarks on the bottom axis.
#
res.tmXBMode             = "Manual"
res.tmXBTickSpacingF     = 2.5
res.tmXBLabelFont        = "Helvetica-Bold"
res.tmXBLabelFontHeightF = 0.015
res.tmXBMajorLengthF     = 0.02
res.tmXBMinorLengthF     = 0.01
res.tmXBMinorPerMajor    = 4

#
# Control range of X and Y axis.
res.trXMaxF              = 20.0
res.trXMinF              = 0.0
res.trYMinF              = 0.0

#
# Width, height, and position of X axis. Every time the
# axis is redrawn, the vpYF resource will be changed to change
# the position.
# 
res.vpXF                 = 0.2
res.vpYF                 = 0.9
res.vpHeightF            = 0.02
res.vpWidthF             = 0.7

xy = Ngl.xy(wks,x,y,res)
Ngl.draw(xy)

#
# Draw 9 different plots demonstrating control of the bottom x-axis 
# tickmark labels using the XBFormat string resource. See the description
# of the Floating Point Format Specification scheme in the HLU reference
# guide to learn about the semantics and syntax of the format string:
#
#    http://www.ncl.ucar.edu/Document/Graphics/format_spec.shtml
#
# There are links to this description in the TickMark reference pages under
# the entries for the format string resources (XBFormat, for example).
#

sres = Ngl.Resources()

sres.vpYF          = 0.9
sres.tiMainString  = "Default format"
sres.tiYAxisString = "0@*+^sg"

Ngl.set_values(xy.base,sres)
Ngl.draw(xy)

sres.vpYF          = 0.8
sres.tiMainString  = "Equal number of significant digits"
sres.tiYAxisString = "0f"
sres.tmXBFormat    = "0f" 

Ngl.set_values(xy,sres)
Ngl.draw(xy)

sres.vpYF          = 0.7
sres.tiMainString  = "No unnecessary zeroes"
sres.tiYAxisString = "f"
sres.tmXBFormat    = "f" 

Ngl.set_values(xy,sres)
Ngl.draw(xy)

sres.vpYF          = 0.6
sres.tiMainString  = "Force decimal point"
sres.tiYAxisString = "#f"
sres.tmXBFormat    = "#f" 

Ngl.set_values(xy,sres)
Ngl.draw(xy)

#
# Note that when the XBFormat string specifies the precision (number
# of significant digits) explicitly (using the '.' conversion field),
# both XBAutoPrecision and XBPrecision are ignored.
#
sres.vpYF          = 0.5
sres.tiMainString  = "4 significant digits for maximum absolute value"
sres.tiYAxisString = "0@;*.4f"
sres.tmXBFormat    = "0@;*.4f" 

Ngl.set_values(xy,sres)
Ngl.draw(xy)

sres.vpYF          = 0.4
sres.tiMainString  = "Zero fill 5 character field"
sres.tiYAxisString = "0@5;*.4f"
sres.tmXBFormat    = "0@5;*.4f" 

Ngl.set_values(xy,sres)
Ngl.draw(xy)

sres.vpYF          = 0.3
sres.tiMainString  = "Field width 7; fill character *; decimal position 4"
sres.tiYAxisString = "&*0@7;*.4~4f" 
sres.tmXBFormat    = "&*0@7;*.4~4f" 

Ngl.set_values(xy,sres)
Ngl.draw(xy)

#
# Note that the tick spacing is set to a larger value because 
# the exponential notation takes up more space.
#
sres.vpYF             = 0.2
sres.tiMainString     = "Exponential format using superscript notation"
sres.tmXBTickSpacingF = 5.0
sres.tiYAxisString    = "0@!;*^se" 
sres.tmXBFormat       = "0@!;*^se" 

Ngl.set_values(xy,sres)
Ngl.draw(xy)

sres.vpYF          = 0.1
sres.tiMainString  = "Exponential format using '**' notation"
sres.tiYAxisString = "0@!;*^ae" 
sres.tmXBFormat    = "0@!;*^ae" 

Ngl.set_values(xy,sres)
Ngl.draw(xy)

# 
# Draw some text strings to label the plots.
#
txres               = Ngl.Resources()
txres.txFont        = "helvetica-bold"
txres.txJust        = "BottomRight"
txres.txFontHeightF = 0.02

Ngl.text_ndc(wks,"Format String",0.19,0.95,txres)

txres.txJust = "BottomCenter"
Ngl.text_ndc(wks,"Resulting tickmark labels",0.55,0.95,txres)

Ngl.frame(wks)

Ngl.end()
