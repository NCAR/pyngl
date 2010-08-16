#
#  File:
#    multi_plot.py
#
#  Synopsis:
#    Illustrates drawing different plots on one page.
#
#  Categories:
#    Multiple contours on one map
#    Polygons
#    Polymarkers
#    Polylines
#    Text
#    XY plot
#
#  Author:
#    Mary Haley (based on original NCL script by Adam Phillips)
#  
#  Date of initial publication:
#    November, 2006
#
#  Description:
#    This example shows how to draw three different kinds of plots
#    on one page, and use polymarkers, lines, polygons, and text
#    to annotate them.
#
#  Effects illustrated:
#    o  Using polygon fill in an XY plot.
#    o  Having multiple scales on an XY plot.
#    o  Using function codes for super/subscripts.
#    o  Overlaying multiple contour plots on a map.
# 
#  Output:
#    This example produces one visualization with three plots:
#      1.)  A bar chart.
#      2.)  A Robinson map projection with two contour plots overlayed.
#      1.)  An XY plot with multiple scales.
#
#  Notes:
#     

#
#  Import numpy and the Masked Array module.
#
import numpy, os
from numpy import ma

#
# Import Ngl and Nio for file I/O and graphics.
#
import Ngl
import Nio

#
#  Open the netCDF files, get variables, add cyclic points.
#
dirc  = Ngl.pynglpath("data")
filea = "ice5g_21k_1deg.nc"
fileb = "sstanom.robinsonproj.nc"

a = Nio.open_file(os.path.join(dirc,"cdf",filea))
b = Nio.open_file(os.path.join(dirc,"cdf",fileb))

topo = a.variables["Topo"][:,:]
Lat  = a.variables["Lat"][:]
Lon  = a.variables["Lon"][:]
icem = a.variables["Icemask"][:,:]

# Keep topo where icem.eq.1
topo = numpy.ma.where(icem == 1, topo,1e20)

# Add longitude cyclic point
topo,Lon = Ngl.add_cyclic(topo,Lon)

sst       = b.variables["SST"][:,:]
lat       = b.variables["lat"][:]
lon       = b.variables["lon"][:]
sst,lon   = Ngl.add_cyclic(sst,lon)


#
# Start the graphics
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"multi_plot")
Ngl.merge_colormaps(wks,"BlWhRe","default")

#========================================================================== 
print "Beginning bar chart section"
arr    = [-2.22,-0.54,-1.4, -3.,-1.2,0.1]
minval = [-2.6, -0.68, -2., -4.,-1.4]
maxval = [-1.87,-0.4,  -1.,-1.9,-1.0]

res = Ngl.Resources()

res.nglMaximize = False
res.nglDraw     = False
res.nglFrame    = False

res.vpWidthF  = 0.4
res.vpHeightF = 0.22
res.vpXF      = .155
res.vpYF      = .97
res.trXMinF   = 0.
res.trXMaxF   = 7.
res.trYMinF   = -4.8
res.trYMaxF   = 1.
        
res.tmXBOn                  = False
res.tmXTOn                  = False
res.tmXBMode                = "Explicit"          # explicit labels
res.tmXBValues              = range(8)
res.tmXBLabels              = ["","","","","","","",""]
res.tmYLLabelFontHeightF    = 0.012
res.tmYLMajorOutwardLengthF = -0.001     # set tickmarks
res.tmYLMinorOutwardLengthF = -0.001
res.tmYLMajorLengthF        = 0.021
res.tmYLPrecision           = 1

res.tiYAxisString      = "Radiative Forcing (Watts per square meter)"
res.tiYAxisFontHeightF = 0.01
res.tiYAxisOffsetXF    = -.017

res.xyLineColor        = "transparent"
           
plot = Ngl.xy(wks,range(1,6),arr,res)

#
# Fill area above Y=0 pink, and below Y=0 light blue.
#
fillres = Ngl.Resources()
fillbox = []

fillres.gsFillColor = "pink"
fillbox.append(Ngl.add_polygon(wks,plot, \
          [res.trXMinF,res.trXMaxF,res.trXMaxF,res.trXMinF,res.trXMinF],
          [0.,0.,res.trYMaxF,res.trYMaxF,0.],fillres))

fillres.gsFillColor = "LightBlue"
fillbox.append(Ngl.add_polygon(wks,plot, \
          [res.trXMinF,res.trXMaxF,res.trXMaxF,res.trXMinF,res.trXMinF],
          [0.,0.,res.trYMinF,res.trYMinF,0.],fillres))

#
# Draw some boxes and bars on the XY plot using calls to Ngl.add_polyline.
#
# First draw a thick line at Y = 0.
#
lineres                  = Ngl.Resources()
lineres.gsLineThicknessF = 3.
zero_line = Ngl.add_polyline(wks,plot,[res.trXMinF,res.trXMaxF],
                                      [0.,0.],lineres)

#
# Add the vertical green "error" bars.
#
time  = Ngl.fspan(1,5,5)
lineres.gsLineThicknessF = 2.5
lineres.gsLineColor      = "DarkGreen"
vlines = []
for ii in range(5):
  ii1 = ii + 1
  vlines.append(Ngl.add_polyline(wks,plot,[ii1,ii1],
                                   [minval[ii],maxval[ii]],lineres))
  vlines.append(Ngl.add_polyline(wks,plot,[ii1-.08,ii1+.08], \
                                   [maxval[ii],maxval[ii]],lineres))
  vlines.append(Ngl.add_polyline(wks,plot,[ii1-.08,ii1+.08], \
                                 [minval[ii],minval[ii]],lineres))

# 
# Draw boxes going up or down from Y=0 line.
#
bars = []
width = 0.15                 # half length of each bar.
for i in range(1,7):
  iw1 = i-width
  iw2 = i+width
  if arr[i-1] <= 0:
    lineres.gsLineColor = "blue"
  else:
    lineres.gsLineColor = "red"

  bars.append(Ngl.add_polyline(wks,plot,[iw1,iw1,iw2,iw2,iw1], \
                                        [0,arr[i-1],arr[i-1],0,0],lineres))

#
# Label boxes with text strings.
#

txres    = Ngl.Resources()
tstrings = []
txres.txFontHeightF = 0.009      
tstrings.append(Ngl.add_text(wks,plot,"CO~B~2~N~",1,-3.,txres))
tstrings.append(Ngl.add_text(wks,plot,"CH~B~4~N~ + N~B~2~N~O",2,-.97,txres))
tstrings.append(Ngl.add_text(wks,plot,"Mineral Dust",3,-2.4,txres))
tstrings.append(Ngl.add_text(wks,plot,"Continental ice",4,-4.35,txres))
tstrings.append(Ngl.add_text(wks,plot,"+ sea level",4,-4.6,txres))
tstrings.append(Ngl.add_text(wks,plot,"Vegetation",5,-1.77,txres))
tstrings.append(Ngl.add_text(wks,plot,"Orbital",6,0.45,txres))

#
# Draw two vertical strings on left side of plot.
#
txres.txFontHeightF = 0.009
txres.txAngleF      = 90.
Ngl.text_ndc(wks,"cooling",0.11,0.835,txres)
Ngl.text_ndc(wks,"warming",0.11,0.95,txres) 

Ngl.draw(plot)    # Draw the plot; all of the attached lines and
                  # text strings will also get drawn.

#========================================================================== 
print "Beginning robinson map projection section"

Ngl.define_colormap(wks,"wh-bl-gr-ye-re")

mpres             = Ngl.Resources()     # New resource list for map plots.

mpres.nglMaximize = False
mpres.nglFrame    = False
mpres.vpWidthF    = 0.5
mpres.vpHeightF   = 0.32
mpres.vpXF        = .11
mpres.vpYF        = .75

mpres.sfXArray          = lon
mpres.sfYArray          = lat

mpres.mpProjection      = "Robinson"        # choose projection
mpres.mpFillOn          = False             # turn on map fill
mpres.mpGridAndLimbOn   = True              # turn on lat/lon/limb lines
mpres.mpGridLineColor   = "transparent"     # we don't want lat/lon lines
mpres.mpPerimOn         = False             # turn off map perimeter
mpres.mpCenterLonF      = 300.

mpres.cnFillOn             = True             # color plot desired
mpres.cnLinesOn            = False            # turn off/on contour lines
mpres.cnLineLabelsOn       = False            # turn off contour lines
mpres.cnLevelSelectionMode = "ManualLevels"   # set manual contour levels
mpres.cnMinLevelValF       = -7.              # set min contour level
mpres.cnMaxLevelValF       =  1.              # set max contour level
mpres.cnLevelSpacingF      =  1.              # set contour spacing
mpres.cnFillColors         = [8,17,30,50,65,95,120,134,152,161]

mpres.pmTickMarkDisplayMode    = "Never"     # Don't draw tickmark border.

mpres.lbOrientation            = "Horizontal"    # put labelbar at bottom
mpres.pmLabelBarSide           = "Bottom"        # and horizontal
mpres.lbTitleString            = "Sea Surface Temperature Change (~S~o~N~C)"
mpres.lbTitlePosition          = "Bottom"
mpres.lbTitleFontHeightF       = 0.015
mpres.lbLabelFontHeightF       = 0.015
mpres.pmLabelBarOrthogonalPosF = 0.025  
        
map = Ngl.contour_map(wks,sst,mpres)      # Draw the first contours/map.

Ngl.define_colormap(wks,"WhViBlGrYeOrRe")   # set colormap for 2nd contours
igray = Ngl.new_color(wks,0.8,0.8,0.8)      # add gray

# Set up new contour resources

mpres.sfXArray          = Lon
mpres.sfYArray          = Lat

mpres.cnMinLevelValF    =  500.           # min contour level
mpres.cnMaxLevelValF    = 3500.           # max contour level
mpres.cnLevelSpacingF   =  500.           # contour spacing
mpres.cnFillColors      = [20,17,14,12,10,8,6,4]

mpres.lbOrientation      = "Vertical"      # vertical labelbar on 
mpres.pmLabelBarSide     = "Left"          # left side
mpres.lbTitleString      = "Ice Sheet Elevation (m)"
mpres.lbTitlePosition    = "Left"
mpres.lbTitleFontHeightF = 0.015
mpres.lbTitleDirection   = "Across"
mpres.lbTitleAngleF      = 90
mpres.lbTitleOffsetF     = 0.3
mpres.pmLabelBarHeightF  = 0.53
mpres.pmLabelBarWidthF   = 0.07
mpres.lbLabelFontHeightF       = 0.012

map2 = Ngl.contour_map(wks,topo,mpres)      # Draw the second contours/map.

#========================================================================== 
print "Beginning new XY plot section"

yarr = numpy.array([[0.,16.],[15.,19.], [22.5,30.5], [15.,15.], [22.5,22.5]])
xarr = numpy.array([[0.,7.5],[0.,8.],   [0.,8.],     [0.,8.],   [0.,8.]])

sres              = Ngl.Resources()
sres.nglDraw      = False
sres.nglFrame     = False
sres.nglMaximize  = False

sres.vpWidthF     = 0.22
sres.vpHeightF    = 0.4
sres.vpXF         = .72
sres.vpYF         = .92

sres.trXMinF      = 0.
sres.trXMaxF      = 8.
sres.trYMinF      = 0.
sres.trYMaxF      = 32.5

sres.tmXBOn          = True
sres.tmXTOn          = False
sres.tmYROn          = False
sres.tmYRBorderOn    = False
sres.tmXTBorderOn    = False
sres.tmXBMode        = "Explicit"              # explicit labels
sres.tmXBValues      = range(9)
sres.tmXBLabels      = sres.tmXBValues
sres.tmYLMode        = "Explicit"              # explicit labels
sres.tmYLValues      = [0,   5,  10, 15, 17.5, 20.,22.5,27.5,32.5]
sres.tmYLLabels      = ["0","5","10","0","2.5","5", "0",  "5","10"]
sres.tmYLMinorValues = Ngl.fspan(2.5,32.5,13)

sres.tmYLPrecision        = 1
sres.tmXBPrecision        = 1
sres.tmYLLabelFontHeightF = 0.012
sres.tmXBLabelFontHeightF = 0.012

sres.tiYAxisString      = "Regional Temperature Change (~S~o~N~C cooling)"
sres.tiYAxisFontHeightF = 0.015
sres.tiYAxisOffsetXF    = -.015
sres.tiXAxisString      = "Global Temperature Change (~S~o~N~C cooling)"
sres.tiXAxisFontHeightF = 0.012
           
sres.xyMonoDashPattern = False
sres.xyDashPatterns    = [2,2,2,0,0]      # five curves
sres.xyMonoLineColor   = True
sres.xyLineThicknessF  = 1.5

xyplot = Ngl.xy(wks,xarr,yarr,sres)    # Draw XY plot

# Set up resource list to add markers to XY plot.
polyres                   = Ngl.Resources()
polyres.gsMarkerIndex     = 16           # polymarker style
polyres.gsMarkerSizeF     = 10.          # polymarker size
        
# Add some gray-filled boxes.
dum2 = []
        
resp = Ngl.Resources()
resp.gsFillColor = "Gray75"
xpts = [0.,0.,8.,8.,0.]
ypts = [17.1,18.9,18.9,17.1,17.1]
Ngl.polygon (wks,xyplot,xpts,ypts,resp)
ypts = [11.,7.,7.,11.,11.]
Ngl.polygon (wks,xyplot,xpts,ypts,resp)
ypts = [25.7,31.8,31.8,25.7,25.7]
Ngl.polygon (wks,xyplot,xpts,ypts,resp)

# Add 3 sets markers to XY plot.

# 1st set
xcoord = [3.12,3.69, 4.54,5.12,3.35]
ycoord = [2.41,7.42,10.84,9.67,7.03]
polyres.gsMarkerColor = "blue"
for gg in range(5):
  Ngl.polymarker(wks,xyplot,xcoord[gg],ycoord[gg],polyres)

# 2nd set
ycoord = [16.16,17.1,16.75,16.91,17.26]

polyres.gsMarkerColor = "red"
for gg in range(5):
  Ngl.polymarker(wks,xyplot,xcoord[gg],ycoord[gg],polyres)
        
# 3rd set
ycoord = [24.9,26.6,29.1,27.7]
xcoord = [3.12,3.69,4.54,5.12]

polyres.gsMarkerColor = "green"
for gg in range(4):
  Ngl.polymarker(wks,xyplot,xcoord[gg],ycoord[gg],polyres)
        
Ngl.draw(xyplot)    
        
# Ngl.draw_ndc_grid(wks)         # For debug purposes.

# 
# Draw a text string on side of plot.
#
txres.txFontHeightF = 0.01
txres.txAngleF      = 90.
Ngl.text_ndc(wks,"Central Antarctica",0.665,0.61,txres)     
Ngl.text_ndc(wks,"Tropical Atlantic", 0.665,0.75,txres)
Ngl.text_ndc(wks,"North Atlantic",    0.665,0.86,txres) 

Ngl.frame(wks)          # Now advance frame, we're done!

Ngl.end()
