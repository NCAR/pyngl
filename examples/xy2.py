#
#  File:
#    xy2.py
#
#  Synopsis:
#    Illustrates how to add primitives and text to an XY plot.
#
#  Categories:
#    paneling
#    xy plots
#    polygons
#    polylines
#    polymarkers
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    January 2006
#
#  Description:
#    This example shows how to attach primitives and text to an XY
#    plot so that when the plot is resized and drawn, the primitives
#    and text are automatically resized and drawn.

#  Effects illustrated:
#    o  Using named colors to create a color map.
#    o  Attaching primitives to a plot.
#    o  Attaching text to a plot.
#    o  Panelling plots.
# 
#  Output:
#     A single visualization with two plots is produced.
#
#  Notes:
#     

#
#  Import NumPy.
#
import numpy

#
#  Import Ngl support functions.
#
import Ngl

PI = 3.14159
x = numpy.arange(1,65,1)       # X points for both XY plots
y = numpy.sin(PI*x/32.)        # Y points for first XY plot

#
# Set up arrays for polygons, polymarkers, and polylines.
#
xx = numpy.array([1.,32.,64.],'f')
ytopline = [ 0.5, 0.5, 0.5]     # Top line
ymidline = [ 0.0, 0.0, 0.0]     # Middle line
ybotline = [-0.5,-0.5,-0.5]     # Bottom line

xdots = x[26:37:5]     # X and Y coordinates for polymarkers.
ydots = y[26:37:5]

xsquare = [16.0,48.0,48.0,16.0,16.0]    # X and Y coordinates
ysquare = [-0.5,-0.5, 0.5, 0.5,-0.5]    # for polygon.

#
# Create a PS workstation, and use named colors to generate
# a colormap.
#
wks_type = "ps"
rlist = Ngl.Resources()
rlist.wkColorMap = ["white","black","blue","hotpink","yellow","green",\
                    "coral4","gray75"]
wks = Ngl.open_wks(wks_type,"xy2",rlist)

#
# Set up common resources for the XY plots.
#
xyres          = Ngl.Resources()
xyres.nglDraw  = False
xyres.nglFrame = False
xyres.trXMaxF  = 64

# Create first XY plot, but don't draw it yet.

xy1 = Ngl.xy(wks,x,y,xyres)  

#
# Set some resources for the primitives to be drawn.
#
gsres                   = Ngl.Resources()

# Polyline resources.

gsres.gsLineColor       = "Blue"
gsres.gsLineThicknessF  = 3.0      # thrice thickness

# Polygon resources.

gsres.gsFillColor       = "Yellow"
gsres.gsEdgesOn         = True    # Line will be drawn around polygon.

# Polymarker resources.

gsres.gsMarkerIndex     = 16       # dots
gsres.gsMarkerColor     = "HotPink"
gsres.gsMarkerSizeF     = 0.014    # twice normal size

#
# Add primitives to plot.  They will be drawn in the order
# they are added, so some primitives may cover other ones.
#
# Be sure to use unique variable names on the left-hand side
# for the Ngl.add_polyxxxx routines.
#
prim1 = Ngl.add_polyline(wks,xy1,xx,ymidline,gsres)
prim2 = Ngl.add_polygon(wks,xy1,xsquare,ysquare,gsres)
prim3 = Ngl.add_polyline(wks,xy1,xx,ytopline,gsres)
prim4 = Ngl.add_polyline(wks,xy1,xx,ybotline,gsres)

gsres.gsLineColor      = "Green"
prim5 = Ngl.add_polyline(wks,xy1,x[26:37],y[26:37],gsres)
prim6 = Ngl.add_polymarker(wks,xy1,xdots,ydots,gsres)

#
# New Y points for second XY plot.
#
y = numpy.cos(3.14159*x/32.)

# Create second XY plot, but don't draw it yet.

xy2 = Ngl.xy(wks,x,y,xyres)

#
# Add primitives to second plot.
#
gsres.gsLineColor      = "Blue"

prim7  = Ngl.add_polyline(wks,xy2,xx,ymidline,gsres)
prim8  = Ngl.add_polygon(wks,xy2,xsquare,ysquare,gsres)
prim9  = Ngl.add_polyline(wks,xy2,xx,ytopline,gsres)
prim10 = Ngl.add_polyline(wks,xy2,xx,ybotline,gsres)

gsres.gsLineColor      = "Green"

xdots2 = x[15:21:5]
ydots2 = y[15:21:5]
prim11 = Ngl.add_polyline(wks,xy2,x[15:21],y[15:21],gsres)
prim12 = Ngl.add_polymarker(wks,xy2,xdots2,ydots2,gsres)

xdots2 = x[42:48:5]
ydots2 = y[42:48:5]
prim13 = Ngl.add_polyline(wks,xy2,x[42:48],y[42:48],gsres)
prim14 = Ngl.add_polymarker(wks,xy2,xdots2,ydots2,gsres)

#
# Now add some text strings.
#
txres               = Ngl.Resources()
txres.txFont        = "helvetica-bold"
txres.txFontColor   = "coral4"
txres.txFontHeightF = 0.037
txres.txJust        = "BottomCenter"

text1  = Ngl.add_text(wks,xy1,"this is a text string",32.,0.5,txres)

#
# Slanted strings
#
txres.txJust    = "BottomRight"
txres.txAngleF  = -45.
text2b = Ngl.add_text(wks,xy2,"some slanted",32.,.5,txres)
txres.txJust    = "BottomLeft"
txres.txAngleF  =  45.
text2b = Ngl.add_text(wks,xy2,"text strings",32.,.5,txres)

#
# Filled text box.
#
txres.txAngleF              = 0.
txres.txJust                = "CenterLeft"
txres.txPerimOn             = True
txres.txFontColor           = "black"
txres.txBackgroundFillColor = "gray75"
text4  = Ngl.add_text(wks,xy1,"filled text box",3.,-1.0,txres)

#
# Now that our primitives have been added to both plots, panel them.
# Note that by adding the primitives, they will get drawn when the
# XY plots are drawn, and they will be automatically resized with
# the plot.
#
Ngl.panel(wks,[xy1,xy2],[2,1])

Ngl.end()
