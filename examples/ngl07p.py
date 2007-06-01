#
#  File:
#    ngl07p.py
#
#  Synopsis:
#    Uses spline interpolation and draws two XY plots on the same frame.
#
#  Categories:
#    Polylines
#    Polymarkers
#    Text
#    XY plots
#    Processing
#
#  Author:
#    Fred Clare
#  
#  Date of initial publication:
#    September, 2004
#
#  Description:
#    This examples uses one-dimensional spline interpolation 
#    for a curve on an XY plot; it finds the integral at each
#    abscissa in the XY plot and draws an XY plot of that on 
#    the same frame.
#
#  Effects illustrated:
#    o  Using functions from the Fitgrid package for interpolation
#       and integration.
#    o  How to draw two XY plots on the same frame.
#    o  How to draw polylines, polymarkers, and text anywhere on a plot.
#    o  Using named colors.
#    o  Many tickmark resource settings.
# 
#  Output:
#    This example produces a single visualization illustrating
#    basic polyline, polymarker, and text capabilities as well as
#    drawing two XY plots on the same frame.
#
#  Notes:
#     

#
#  Import numpy.
#
import numpy

#
#  Import Ngl support functions.
#
import Ngl

xi = [  0.000, 0.210, 0.360, 0.540, 1.000, 1.500, 1.970, 2.300, \
        2.500, 2.700  ]
yi = [  0.000, 2.600, 3.000, 2.500, 0.000,-1.000, 0.000, 0.800, \
        0.920, 0.700  ]

npts   = 201
xl     =  -1.0
xr     =   5.0
inc    = (xr-xl)/(npts-1)
xo     = numpy.arange(xl,xr+0.0001,inc) # Create output X coordinate array.
                                        # Add a small amount to xr to make
                                        # sure xr is the final value in the
                                        # range.
period = 3.0
yo = Ngl.ftcurvp(xi, yi, period, xo)  # Interpolate.

yint = numpy.zeros(npts,'f')
for i in xrange(npts):
  yint[i] = Ngl.ftcurvpi(0., xo[i], period, xi, yi)

cmap = ["white","black","red","green","blue","yellow"]
rlist = Ngl.Resources()
rlist.wkColorMap = cmap
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"ngl07p",rlist) # Open an X11 workstation.

txres = Ngl.Resources()   # Set up variable for TextItem resources.
xyres = Ngl.Resources()   # Set up variable for XyPlot resources.
gsres = Ngl.Resources()   # Set up variable for GraphicStyle resources.

xyres.nglFrame               = False     # Don't advance the frame.
xyres.nglMaximize            = False     # Don't advance the frame.

xyres.tmXTBorderOn           = False     # Don't draw top axis.
xyres.tmXTOn                 = False     # Don't draw top axis tick marks.
xyres.tmBorderThicknessF     = 1.0       # Default thickness is 2.0

xyres.tmXBLabelFont          = 21        # Change font and size of
xyres.tmXBLabelFontHeightF   = 0.025     # X axis labels.
xyres.tmXBMajorLengthF       = 0.015     # Default is 0.02.
xyres.tmXBMajorThicknessF    = 1.0       # Default is 2.
xyres.tmXBMinorLengthF       = 0.0075    # Default is 0.01.
xyres.tmXBMinorPerMajor      = 4         # # of minor tick marks per major.
xyres.tmXBMode               = "Manual"  # Set tick mark mode.
xyres.tmXBTickStartF         = -1.0
xyres.tmXBTickEndF           = 5.0
xyres.tmXBTickSpacingF       = 1.0
xyres.tmXMajorGridThicknessF = 1.0       # Default is 2.0

xyres.tmYLLabelFont          = 21        # See explanations for X axes
xyres.tmYLLabelFontHeightF   = 0.025     # resources.
xyres.tmYLMajorLengthF       = 0.015
xyres.tmYLMajorThicknessF    = 1.0
xyres.tmYLMinorPerMajor      = 0
xyres.tmYLMode               = "Manual"
xyres.tmYLTickStartF         = -2.0
xyres.tmYLTickEndF           = 3.0
xyres.tmYLTickSpacingF       = 1.0
xyres.tmYRBorderOn           = False    # Don't draw right axis.
xyres.tmYROn                 = False    # Don't draw right axis tick marks.

xyres.trXMaxF   = xr  # Specify data limits for X and Y axes.
xyres.trXMinF   = xl
xyres.trYMaxF   =  3.02
xyres.trYMinF   = -2.00

xyres.vpHeightF = 0.25 # Define height, width, and location of plot.
xyres.vpWidthF  = 0.80
xyres.vpXF      = 0.13
xyres.vpYF      = 0.85

xy = Ngl.xy(wks,xo,yo,xyres) # Plot the interpolated values.

xyres.xyMarkLineMode = "Markers"  # Set line mode to "Markers".
xyres.xyMarkerColor  = "blue"     # Set markers to blue.
xyres.xyMarkerSizeF  = 0.025      # Make markers larger.

xy = Ngl.xy(wks,xi,yi,xyres) # Plot the original points with markers.

txres.txFont        = 21                  # Change the default font.
txres.txFontHeightF = 0.03                # Set the font height.
Ngl.text(wks,xy,"Function",1.5,2.5,txres) # Label the plot.

xx = [xl,xr]   # Create data for a polyline for marking
yy = [0.0,0.0] # the Y = 0.0 line in each graph.

gsres.gsLineColor = "red"            # Set polyline color to red.
Ngl.polyline(wks,xy,xx,yy,gsres)     # Draw polyline at Y=0.

#----------- Begin second plot, same frame--------------------------------

xyres.vpYF             = 0.37    # Set Y location of plot.

xyres.trYMinF          = -1.0    # Set minimum Y axis value.
xyres.trYMaxF          =  4.0    # Set maximum Y axis value.

xyres.tmYLTickStartF   = -1.0    # Define tick mark spacing
xyres.tmYLTickSpacingF =  1.0    # along Y axis.
xyres.tmYLTickEndF     =  4.0

xyres.xyMarkLineMode   = "Lines" # Set line mode to "Lines".

xy = Ngl.xy(wks,xo,yint,xyres)  # Plot the integrals.

txres.txFontHeightF = 0.03                              # Set font height.
Ngl.text(wks,xy,"Integral (from X = 0.)",0.8,3.5,txres) # Label plot.

Ngl.polyline(wks,xy,xx,yy,gsres) # Draw polyline at Y=0.

txres.txFontHeightF = 0.04  # Change the font height.

Ngl.text_ndc(wks,"Demo for ftcurvp, ftcurvpi",.5,.95,txres)

xperiod1 = [.41,   .2633, .2633, .2633, .2833, .2633, .2833, .2633, .2633]
xperiod2 = [.5166, .6633, .6633, .6633, .6433, .6633, .6433, .6633, .6633]
yperiod  = [.503,  .503,  .523,  .503,  .513,  .503,  .493,  .503,  .4830]

Ngl.polyline_ndc(wks,xperiod1,yperiod,gsres)  # Draw a period legend.
Ngl.polyline_ndc(wks,xperiod2,yperiod,gsres)  # between the two plots.

txres.txFontHeightF = 0.024                # Set font height.
Ngl.text_ndc(wks,"Period",0.465,0.5,txres) # Label the period legend.

Ngl.frame(wks)        # Advance the frame.

del xy        # Clean up.
del txres
del gsres
del xyres

Ngl.end()
