#
#  File:
#    wmstnm02.py
#
#  Synopsis:
#    Illustrates plotting station model data and wind barbs.
#
#  Category:
#    Wind barbs/Station model data.
#
#  Author:
#    Fred Clare
#
#  Date of initial publication:
#    September, 2007 
#
#  Description:
#    Draws a single frame with four plots on it illustrating
#    drawing windbarbs and station model data.
#
#  Effects illustrated:
#    o  Drawing station model data over a map.
#    o  Drawing wind barbs over a map.
#    o  How to panel four wind barb/station model data plots.
#    o  How to draw a single country from a world map.
# 
#  Output:
#    A single visualization is produced that contains four plots:
#       o  Wind barbs over a map.
#       o  A table of wind barbs and related speeds.
#       o  Station model data for some U.S. cities on a map.
#       o  Wind barbs drawn over a map of France.
#
#  Notes:
#     

import numpy
import Ngl

#
#  The procedure tricolour produces a French flag using the
#  official colors and aspect ratio.  The lower left corner
#  of the flag is at coordinate (xll,yll) in NDC.
#
def tricolour(wks,xll,yll,height):
#
#  Colors
#
  blue_index  = Ngl.new_color(wks, 0.00, 0.15, 0.50)
  white_index = Ngl.new_color(wks, 1.00, 1.00, 1.00)
  red_index   = Ngl.new_color(wks, 1.00, 0.10, 0.14)
  width = 1.5*height
  xx = numpy.zeros(5,'f')
  yy = numpy.zeros(5,'f')
#
#  Draw blue bar.
#
  xx = [xll, xll + width/3., xll + width/3., xll, xll]
  yy = [yll, yll, yll+height, yll+height, yll]
  pres = Ngl.Resources()
  pres.gsFillColor = blue_index
  Ngl.polygon_ndc(wks,xx,yy,pres)
#
#  Draw white bar.
#
  xx = [xll+width/3., xll + 2.*width/3., xll + 2.*width/3., \
           xll+width/3., xll+width/3.]
  pres.gsFillColor = white_index
  Ngl.polygon_ndc(wks,xx,yy,pres)
#
#  Draw red bar.
#
  xx = [xll+2.*width/3., xll + width, xll + width,  \
           xll+2.*width/3., xll+2.*width/3.]
  pres.gsFillColor = red_index
  Ngl.polygon_ndc(wks,xx,yy,pres)
  xx = [xll, xll + width, xll + width, xll, xll]
#
#  Perimeter.
#
  pres.gsLineColor = 1
  Ngl.polyline_ndc(wks,xx,yy,pres)

#
#  Show how to put four plots, illustrating wind barb procedures,
#  on a single frame.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"wmstnm02")
cmap = numpy.array([[1.,1.,1.],[0.,0.,0.],[1.,0.,0.]])  # color map

Ngl.define_colormap(wks,cmap)

#############################
#                           #
#  Lower left quatrant.     #
#                           #
#############################
#
#  Use the procedure wmbarb to produce a table of the association
#  of wind barb symbols with wind speeds.  This plot simply uses
#  NDC coordinates - the three other plots show how to plot
#  wind barbs over maps and position those plots on the frame.
#
width  = 0.42
height = 0.42
xl = 0.06
xr = xl + width
yt = 0.46
yb = yt - height
midx = 0.5*(xl+xr)
#
#  Perimeter
#
Ngl.polyline_ndc(wks,[xl,xr,xr,xl,xl],[yb,yb,yt,yt,yb])
#
#  Title
#
txres = Ngl.Resources()
txres.txFont = "Helvetica-Bold"
txres.txFontHeightF = 0.042*height
Ngl.text_ndc(wks,"Wind Speeds",midx,yt+1.3*txres.txFontHeightF,txres)

#
#  Dividing lines.
#
hline_y = yt-0.08*height
Ngl.polyline_ndc(wks,[xl,xr],[hline_y, hline_y],False)
Ngl.polyline_ndc(wks,[midx,midx],[yb, yt],False)

#
#  Row labels.
#
txres.txFont = "Helvetica"
txres.txFontHeightF = 0.0315*height
Ngl.text_ndc(wks,"Symbol  Knots  Miles/hr.",  \
               0.5*(xl+midx),yt-0.04*height,txres)
Ngl.text_ndc(wks,"Symbol  Knots  Miles/hr.",  \
               midx + 0.5*(xr-midx),yt-0.04*height,txres)
#
#  Increase the size of the barbs.
#
Ngl.wmsetp("wbs", .035)
#
#  Left column of table.
#
speeds_u = [ 0., -1.5, -4., -9., -14., -19., -24., -29., -34.]
labels_k = ["Calm","1-2","3-7","8-12","13-17","18-22",  \
             "23-27","28-32","33-37"]
labels_m = ["Calm","1-2","3-8","9-14","15-20","21-25",  \
             "26-31","32-37","38-43"]
y_start = hline_y + 0.04*height
for i in xrange (1,10):
  xp = xl+0.25*(midx-xl)
  yp = y_start - 0.042*i
  Ngl.wmbarb(wks, xp, yp, speeds_u[i-1], 0.)  # Plot barbs.
  Ngl.text_ndc(wks,labels_k[i-1],xp+0.21*(midx-xl),yp,txres) 
  Ngl.text_ndc(wks,labels_m[i-1],xp+0.55*(midx-xl),yp,txres) 
#
#  Right column of table.
#
speeds_u = [ -39., -44., -49., -54., -59., -64., -69., -74., -104.]
labels_k = ["38-42","43-47","48-52","53-57","58-62",  \
             "63-67","68-72","73-77","103-107"]
labels_m = ["44-49","50-54","55-60","61-66","67-71",  \
             "72-77","78-83","84-89","119-123"]
y_start = hline_y+0.04*height
for i in xrange(1,10):
  xp = midx+0.25*(xr-midx)
  yp = y_start - 0.042*i
  Ngl.wmbarb(wks, xp, yp, speeds_u[i-1], 0.)  # Plot barbs.
  Ngl.text_ndc(wks,labels_k[i-1],xp+0.21*(midx-xl),yp,txres) 
  Ngl.text_ndc(wks,labels_m[i-1],xp+0.55*(midx-xl),yp,txres) 

#############################
#                           #
#  Upper left quatrant.     #
#                           #
#############################
#
#  Draw an orthographic map centered at lat/lon = (70.,-10.).
#  Don't advance the frame, since we want to add wind barbs.
#
mpres              = Ngl.Resources()
#
# Define size of map in frame.
#
mpres.mpShapeMode = "FreeAspect"
mpres.vpXF        = 0.06
mpres.vpYF        = 0.94
mpres.vpWidthF    = 0.42
mpres.vpHeightF   = 0.42

mpres.nglFrame     = False
gray_index = Ngl.new_color(wks,0.7,0.7,0.7)
cyan_index = Ngl.new_color(wks,0.5,0.9,0.9)
mpres.mpFillAreaSpecifiers  = ["Water","Land"]
mpres.mpSpecifiedFillColors = [cyan_index,gray_index]

mpres.mpLimitMode  = "LatLon"
mpres.mpMinLonF    = -40.
mpres.mpMaxLonF    =  20.
mpres.mpMinLatF    =  55.
mpres.mpMaxLatF    =  85.
mpres.mpCenterLatF =  70.
mpres.mpCenterLonF = -10.
mpres.mpPerimOn    = True
mpres.mpProjection = "Orthographic"
mpres.tiMainString = "Winds from the south"
mpres.tiMainFont   = "Helvetica-Bold"

mpres.mpFillOn = True
mpres.nglMaximize   = False
mpres.pmTickMarkDisplayMode = "NoCreate"
map = Ngl.map(wks,res=mpres)

#
#  Draw an array of vertical wind barbs over the above map.
#
Ngl.wmsetp("col", 2)     # Draw in red.
Ngl.wmsetp("wbs", .035)  # Increase the size of the barb.

#
#  For illustration use numpy arrays for storing the wind 
#  barb information for the wmbarbmap call (could more easily
#  make individual calls in a loop in the case here).
#
lat =     numpy.zeros([3,10],'f')
lon =     numpy.zeros([3,10],'f')
u   =  0.*numpy.ones([3,10],'f')
v   = 90.*numpy.ones([3,10],'f')
lat[0,:] = 65.
lat[1,:] = 70.
lat[2,:] = 75.
lonp = Ngl.fspan(-40.,5.,10)
lon[0,:] = lonp
lon[1,:] = lonp
lon[2,:] = lonp
Ngl.wmbarbmap(wks,lat,lon,u,v)

#############################
#                           #
#  Upper right quadrant.    #
#                           #
#############################
#
#  Plot some station model data for some cities in
#  Kansas and Oklahoma.
#
# Change location of map in frame. Size is the same as before.
#
mpres.vpXF         =   0.53
mpres.vpYF         =   0.94
mpres.mpLimitMode  = "LatLon"
mpres.mpMinLonF    = -103.0
mpres.mpMaxLonF    =  -94.0
mpres.mpMinLatF    =   32.5
mpres.mpMaxLatF    =   41.5
mpres.mpCenterLatF =   35.0
mpres.mpCenterLonF =  -97.0
mpres.mpPerimOn    = True
mpres.mpOutlineBoundarySets   = "USStates"
mpres.mpUSStateLineThicknessF = 1.25

mpres.tiMainString = "Kansas and Oklahoma stations"
mpres.tiMainFont   = "Helvetica-Bold"
mpres.mpFillOn = False
mpres.mpProjection = "Orthographic"
map = Ngl.map(wks,res=mpres)

#
#  Flag the station model procedure that the wind barbs are
#  being drawn over a map.
#
Ngl.wmsetp("ezf",1)
#
#  Draw in the foreground color.
#
Ngl.wmsetp("col", 1)

#
#  Plot station model data at selected cities.
#
cities = ["Kansas City"  , "Tulsa"  , "Oklahoma City", "Wichita",   \
          "Woodward, OK" , "Goodland, KS"            , "Hays, KS"]
city_lats = [ 39.10,  36.13,  35.48,  37.69,  36.43,   39.35,  38.88]
city_lons = [-94.67, -95.94, -97.53, -97.34, -99.40, -101.71, -99.32]
imdat = ["11721700181008020050300004983052026604007289086925",  \
         "11060032571033020380300004033056030610507808089258",  \
         "11854813511029020330300004025054016609507726087036",  \
         "11000022751126021360300004955054054600007757087712",  \
         "11515500121004020000300004975050034603017207084703",  \
         "11751718481027020310300004021053012609017685086925",  \
         "11206227031102021040300004963056046601517084081470"]
Ngl.wmstnm(wks,city_lats,city_lons,imdat)

#############################
#                           #
#  Lower right quatrant.    #
#                           #
#############################
#                           
#  Draw some wind barbs at cities in France.  Color the
#  country with the blue color from the Franch flag and
#  the wind barbs with the red color from the French flag.
#  
#  Change location of map in frame. Size is the same.
#  
mpres.vpXF         = 0.53
mpres.vpYF         = 0.46
mpres.mpLimitMode  = "LatLon"
mpres.mpMinLonF    = -4.25
mpres.mpMaxLonF    =  8.25
mpres.mpMinLatF    =  39.5
mpres.mpMaxLatF    =  52.0
mpres.mpCenterLatF =  49.
mpres.mpCenterLonF =  2.
mpres.mpPerimOn    = True
mpres.mpFillOn     = True

mpres.mpOutlineBoundarySets  = "NoBoundaries"
mpres.mpDataBaseVersion      = "MediumRes"
mpres.tiMainString           = "Winds Of France"
mpres.tiMainFont             = "Helvetica-Bold"
mpres.mpAreaMaskingOn        = True
mpres.mpFillBoundarySets     = "NoBoundaries"
mpres.mpGridAndLimbOn        = False
mpres.mpAreaMaskingOn        = 1
#
#  Mask countries bordering France.
#
mpres.mpMaskAreaSpecifiers   = ["Water","Ocean","Scotland","Netherlands", \
                                "Belgium","Luxembourg","Italy","Corsica", \
                                "United Kingdom","Germany","Spain",       \
                                "Switzerland"]
mpres.mpFillAreaSpecifiers = ["Land"]
blue_index = Ngl.new_color(wks,0.,0.15,0.5)
red_index  = Ngl.new_color(wks,1.0,0.10,0.14)
mpres.mpSpecifiedFillColors = [blue_index]
map = Ngl.map(wks,"Satellite",mpres)

Ngl.wmsetp("wbs", .04)       # Increase the size of the barb.
Ngl.wmsetp("col", red_index) # Color in red.

#
#  Plot wind barbs at various French cities.
#
#
cities = ["Paris", "LeHavre", "Renne", "Tour", "Dijon", "Clarmont-Ferrand", \
          "Bordeaux", "Toulouse", "Lyon", "Nice", "Nancy", "Limoges",       \
          "Nantes", "Lille", "Morlaix", "Reims", "La Rochelle", "Redon"]
city_lats = [ 48.87, 49.50, 48.10, 47.30, 47.30, 45.76,   \
              44.83, 43.60, 45.75, 43.70, 48.70, 45.83,   \
              47.20, 50.60, 48.58, 49.30, 46.10, 47.67]
city_lons = [  2.33,  0.12,  1.67,  0.68,  5.03,  3.10,   \
              -0.57,  1.45,  4.85,  7.28,  6.15,  1.26,   \
              -1.55,  3.10, -3.83,  4.03,  -1.0, -2.30]
u_compnts = [ -10.,  -5., -30., -20., -15., -15.,  \
              -20.,  -5., -10.,  -5., -20., -30.,  \
              -15., -10., -10., -40., -30., -30.]
v_compnts = [ 2.,  0.,  6.,  4.,  5., 13.,  \
             25., 10., 15., 20., 10., 40.,  \
              5.,  1.,  0.,  8., 30.,  0.]

Ngl.wmbarbmap(wks, city_lats, city_lons, u_compnts, v_compnts)

#
#  Add a French flag.
#
tricolour(wks, 0.84, 0.07, 0.05)

Ngl.frame(wks)

Ngl.end()
