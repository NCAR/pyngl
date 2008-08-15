#
#  File:
#    ctrydiv1.py
#
#  Synopsis:
#    Draws provinces of China and states of India
#
#  Categories:
#    maps only
#
#  Author:
#    Mary Haley (based on an NCL script)
#  
#  Date of initial publication:
#    August, 2008
#
#  Description:
#    This example uses the new Earth..4 map database to draw
#    the provinces of China and the states of India.
#
#  Effects illustrated:
#      o  How to select a map database resolution.
#      o  How to generate both outlines and filled areas
#         of a specific map outline.
# 
#  Output:
#     A visualization with two frames is produced.
#

#
#  Import Ngl support functions.
#
import Ngl

#
# Qualifiers for China provinces and India states.
#
boundaries = ["China:states","India:States"]

#
# Change the color map.
#
wks_type = "ps"
wks = Ngl.open_wks (wks_type,"ctrydiv1")
Ngl.define_colormap(wks,"StepSeq25")

#
#  Create the plot.
#

#
#  Map resources.
#
res = Ngl.Resources()

res.mpDataSetName         = "Earth..4"   # Change database
res.mpDataBaseVersion     = "MediumRes"  # Medium resolution database

res.mpOutlineOn           = True         # Turn on map outlines
res.mpOutlineBoundarySets = "National"
res.mpOutlineSpecifiers   = boundaries

res.mpLimitMode           = "LatLon"        # Limit map via lat/lon
res.mpGridMaskMode        = "MaskNotOcean"  # Draw grid over ocean.
res.mpOceanFillColor      = -1
res.mpGridLineDashPattern = 7

res.tiMainFont            = "helvetica"
res.tiMainFontHeightF     = 0.02

res.tiMainString          = "Provinces of China"
res.mpMinLatF             =  17
res.mpMaxLatF             =  55
res.mpMinLonF             =  72
res.mpMaxLonF             = 136

plot = Ngl.map(wks,res)                  # Create the map plot

res.mpFillOn              = True         # Turn on map fill
res.mpFillBoundarySets    = "National"
res.mpFillAreaSpecifiers  = boundaries

res.tiMainString          = "States of India"
res.mpMinLatF             =   6
res.mpMaxLatF             =  38
res.mpMinLonF             =  65
res.mpMaxLonF             = 100

Ngl.map(wks,res)

Ngl.end()
