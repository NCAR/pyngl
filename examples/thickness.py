#
#  File:
#    thickness.py
#
#  Synopsis:
#    Draws polylines using various line thicknesses.
#
#  Categories:
#    polylines
#
#  Author:
#    Fred Clare
#
#  Date of initial publication:
#    March, 2006
#
#  Description:
#    This example shows how to change line thicknesses for
#    polyline draws and can be used to illustrate how you
#    can get significantly different results depending on
#    the output devive.
#
#  Effects illustrated:
#    o  Drawing polylines.
#    o  Changing polyline thicknesses.
#
#  Output:
#     A single visualization is produced.
#
#  Notes:
#

import Ngl

rlist = Ngl.Resources()
rlist.wkBackgroundColor = "White"
rlist.wkForegroundColor = "Black"
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"thickness",rlist)

#
#  Main title.
#
txres = Ngl.Resources()
txres.txFontHeightF = 0.04
txres.txFont = 22
Ngl.text_ndc(wks,"Line thicknesses",0.5,0.95,txres)
txres.txFontHeightF = 0.025
Ngl.text_ndc(wks, \
   "(results will vary depending on the output device)",0.5,0.88,txres)

#
#  Specify the sample line thicknesses.
#
thicknesses = [8.00, 4.00, 2.00, 1.00, 0.75, 0.50, 0.25, 0.00]

#
#  Draw and label lines with the sample thicknesses.
#
pres = Ngl.Resources()
ytop = 0.78
txres.txFontHeightF = 0.02
txres.txJust = "CenterLeft"
for i in xrange(len(thicknesses)):
  pres.gsLineThicknessF  = thicknesses[i]
  yh = ytop - 0.1*i
  Ngl.text_ndc(wks,"Thickness = " + str(thicknesses[i]),0.05,yh+0.025,txres)
  Ngl.polyline_ndc(wks,[0.05,0.95],[yh,yh],pres)

Ngl.frame(wks)
Ngl.end()
