#
#  File:
#    map2.py
#
#  Synopsis:
#    Illustrates how to to draw polygons, lines, and markers on a map.
#
#  Categories:
#    maps only
#    polygons
#    polylines
#    polymarkers
#
#  Author:
#    Mary Haley (based on an NCL script of Dave Brown, SCD, NCAR)
#  
#  Date of initial publication:
#    January, 2006
#
#  Description:
#    This example shows how to draw polygons, markers, and lines on top
#    of a map.
#
#  Effects illustrated:
#    o  Using an Orthographic map projection.
#    o  Changing various primitive resources.
# 
#  Output:
#     A single visualization is produced.
#
#  Notes:
#     

import Ngl

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"map2")
 
mpres = Ngl.Resources()

mpres.nglFrame     = False         # Don't advance frame after plot is drawn.
mpres.nglDraw      = False         # Don't draw plot just yet.

mpres.mpProjection = "Orthographic"
mpres.mpCenterLatF = 30.
mpres.mpCenterLonF = 160.
mpres.mpCenterRotF = 60.
mpres.mpLabelsOn   = False
mpres.mpPerimOn    = False

map = Ngl.map(wks,mpres)         # Create map.

gsres = Ngl.Resources()

gsres.gsLineColor            = "black"
gsres.gsLineThicknessF       = 2.0
gsres.gsLineLabelFont        = "helvetica-bold"
gsres.gsLineLabelFontHeightF = 0.02
gsres.gsLineLabelFontColor   = "black"
gsres.gsFillIndex            = 0
gsres.gsEdgesOn              = False

# Tropical zone

gsres.gsFillColor = 87

px = [360.,360.,0.,0.,360.]
py = [-23.5,23.5,23.5,-23.5,-23.5]

Ngl.polygon(wks,map,px,py,gsres)

# North and South temperate zones.

gsres.gsFillColor = 135

py[0:5] = [ 23.5 , 66.5 , 66.5 , 23.5 , 23.5]
px[0:5] = [ 360. , 360. , 0. , 0. , 360.]
Ngl.polygon(wks,map,px,py,gsres)

py[0:5] = [ -23.5 , -66.5 , -66.5 , -23.5 , -23.5 ]
Ngl.polygon(wks,map,px,py,gsres)

# Frigid zones.

gsres.gsFillColor = 167

py[0:5] = [ 90. , 66.5 , 66.5 , 90. , 90. ]
Ngl.polygon(wks,map,px,py,gsres)

py[0:5] = [ -90. , -66.5 , -66.5 , -90. , -90.]
Ngl.polygon(wks,map,px,py,gsres)

# Markers at each pole.

gsres.gsMarkerIndex          = 15      # A hollow circle with an "X"
gsres.gsMarkerSizeF          = 0.02
gsres.gsMarkerThicknessF     = 2.0
gsres.gsMarkerColor          = "background"

px[0:2] = [ 0.,   0.]
py[0:2] = [90., -90.]
Ngl.polymarker(wks,map,px[:2],py[:2],gsres)

#
# Draw polylines at each of the major latitudinal boundary lines,
# beginning with the equator. Use the line label to name each of the
# lines. The '|' character is inserted between each label character 
# to allow the labels to track the curve of each line more precisely.
#

gsres.gsLineLabelString = "e|q|u|a|t|o|r"

px[0:2] = [ 360.,  0.]
py[0:2] = [   0.,  0.]
Ngl.polyline(wks,map,px[:2],py[:2],gsres)

# Tropic of cancer.

gsres.gsLineLabelString = "t|r|o|p|i|c o|f c|a|n|c|e|r"

px[0:2] = [360.,  0.]
py[0:2] = [23.5, 23.5]
Ngl.polyline(wks,map,px[:2],py[:2],gsres)

#
# Tropic of capricorn. Note: currently there is a limit on the 
# number of characters in a line label that prevents the '|'
# character from being used between each letter in a label 
# of this length.
#
gsres.gsLineLabelString = "tr|o|p|ic of c|a|p|r|i|c|o|rn"

py[0:2] = [-23.5, -23.5]
Ngl.polyline(wks,map,px[:2],py[:2],gsres)

# Arctic circle

gsres.gsLineLabelString = "a|r|c|t|i|c c|i|r|c|l|e"

py[0:2] = [66.5, 66.5]
Ngl.polyline(wks,map,px[:2],py[:2],gsres)

# Antarctic circle

gsres.gsLineLabelString = "|a|n|t|a|r|c|t|i|c c|i|r|c|l|e"

py[0:2] = [-66.5, -66.5]
Ngl.polyline(wks,map,px[:2],py[:2],gsres)

Ngl.draw(map)                         # Now draw the map and 
Ngl.frame(wks)                        # advance the frame.

Ngl.end()
