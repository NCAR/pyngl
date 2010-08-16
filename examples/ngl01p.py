#
#  File:
#    ngl01p.py
#
#  Synopsis:
#    Introductory tutorial example using resources for XY plots.
#
#  Category:
#    XY plots
#
#  Author:
#    Fred Clare
#  
#  Date of initial publication:
#    September, 2004
#
#  Description:
#    This is the first of the tutorial examples and is meant to
#    show PyNGL basics by progressing through successively more
#    complicated XY visualizations.
#
#  Effects illustrated:
#    o  How to begin and end a PyNGL script.
#    o  How to create and initialize variables
#    o  How to create and draw XY visualizations.
#    o  How to set resources to change the appearance of a visualization.
#    o  How to read data from an ASCII file.
#    o  How to title a visualization with a main title.
#    o  How to title the X and Y axes of an XY visualization.
#    o  How to use markers in XY visualizations.
#    o  How to change line thicknesses.
#
#  Output:
#    Five visualizations are produced:
#      1.)  A simple XY plot with one curve.
#      2.)  A simple XY plot with three curves and titling of X/Y axes.
#      3.)  As in 2.) but changing the colors of the curves and
#           the curve thicknesses.
#      4.)  Same as 3.), but with a main title and using markers.
#      5.)  XY plot with user specified labels for the curves.
#
#  Notes:
#     

#
#  Import numpy.
#
import numpy, os

#
#  Import Ngl support functions.
#
import Ngl

#
#  Define coordinate data for an XY plot.
#
x = [10., 20.00, 30., 40.0, 50.000, 60.00, 70., 80.00, 90.000]
y = [ 0.,  0.71,  1.,  0.7,  0.002, -0.71, -1., -0.71, -0.003]

wks_type = "ps"
wks = Ngl.open_wks(wks_type,"ngl01p")  # Open a workstation.

plot = Ngl.xy(wks,x,y)   # Draw an XY plot.
  
#
#  Define three curves for an XY plot.  Any arrays in PyNGL codes
#  that are not 1-dimensional must be numpy.arrays.  A 1-dimensional
#  array can be a Python list, a Python tuple, or a numpy.array.
#
#----------- Begin second plot -----------------------------------------
  
y2 = numpy.array([[0., 0.7, 1., 0.7, 0., -0.7, -1., -0.7, 0.],  \
                  [2., 2.7, 3., 2.7, 2.,  1.3,  1.,  1.3, 2.],  \
                  [4., 4.7, 5., 4.7, 4.,  3.3,  3.,  3.3, 4.]], \
                  'f')
  
#
#  Set resources for titling.
#
resources = Ngl.Resources()
resources.tiXAxisString = "X"
resources.tiYAxisString = "Y"
  
plot = Ngl.xy(wks,x,y2,resources)    # Draw the plot.
  
#----------- Begin third plot -----------------------------------------
  
resources.xyLineColors        = [189,107,24]  # Define line colors.
resources.xyLineThicknesses   = [1.,2.,5.]    # Define line thicknesses
                                              # (1.0 is the default).
  
plot = Ngl.xy(wks,x,y2,resources)         
  
#---------- Begin fourth plot ------------------------------------------
  
resources.tiMainString    = "X-Y plot"  # Title for the XY plot
resources.tiXAxisString   = "X Axis"    # Label for the X axis
resources.tiYAxisString   = "Y Axis"    # Label for the Y axis
resources.tiMainFont      = "Helvetica" # Font for title
resources.tiXAxisFont     = "Helvetica" # Font for X axis label
resources.tiYAxisFont     = "Helvetica" # Font for Y axis label
  
resources.xyMarkLineModes = ["Lines","Markers","MarkLines"]
resources.xyMarkers       = [0,1,3]     # (none, dot, asterisk)
resources.xyMarkerColor   = 107         # Marker color
resources.xyMarkerSizeF   = 0.03        # Marker size (default
                                          # is 0.01)
  
plot = Ngl.xy(wks,x,y2,resources)       # Draw an XY plot.
  
#---------- Begin fifth plot ------------------------------------------
  
filename = os.path.join(Ngl.pynglpath("data"),"asc","xy.asc")
data     = Ngl.asciiread(filename,(129,4),"float")
  
#
#  Define a two-dimensional array of data values based on
#  columns two and three of the input data.
#
uv = numpy.zeros((2,129),'f')
uv[0,:] = data[:,1]
uv[1,:] = data[:,2]
  
#
#  Use the first column of the input data file (which is simply the
#  row number) for longitude values.  The fourth column of data is
#  not used in this example.
#
lon     = data[:,0]
lon     = (lon-1) * 360./128.
  
del resources            # Start with new list of resources.
resources = Ngl.Resources()

resources.tiMainString           = "U/V components of wind"
resources.tiXAxisString          = "longitude"
resources.tiYAxisString          = "m/s"
resources.tiXAxisFontHeightF     = 0.02        # Change the font size.
resources.tiYAxisFontHeightF     = 0.02
  
resources.xyLineColors           = [107,24]       # Set the line colors.
resources.xyLineThicknessF       = 2.0         # Double the width.
  
resources.xyLabelMode            = "Custom"    # Label XY curves.
resources.xyExplicitLabels       = ["U","V"]   # Labels for curves
resources.xyLineLabelFontHeightF = 0.02        # Font size and color
resources.xyLineLabelFontColor   = 189         # for line labels
  
plot = Ngl.xy(wks,lon,uv,resources) # Draw an XY plot with 2 curves.

# Clean up.
del plot 
del resources

Ngl.end()
