#
#  Import NumPy.
#
import Numeric

#
#  Import Ngl support functions.
#
from Ngl import *

#
#  Define coordinate data for an XY plot.
#
x = [10., 20.00, 30., 40.0, 50.000, 60.00, 70., 80.00, 90.000]
y = [ 0.,  0.71,  1.,  0.7,  0.002, -0.71, -1., -0.71, -0.003]

wks = ngl_open_wks("pdf","ngl01p")  # Open a workstation.

plot = ngl_xy(wks,x,y)   # Draw an XY plot.
  
#
#  Define three curves for an XY plot.  Any arrays in PyNGL codes
#  that are not 1-dimensional must be Numeric arrays.  A 1-dimensional
#  array can be a Python list, a Python tuple, or a Numeric array.
#
#----------- Begin second plot -----------------------------------------
  
y2 = Numeric.array([[0., 0.7, 1., 0.7, 0., -0.7, -1., -0.7, 0.],  \
                    [2., 2.7, 3., 2.7, 2.,  1.3,  1.,  1.3, 2.],  \
                    [4., 4.7, 5., 4.7, 4.,  3.3,  3.,  3.3, 4.]], \
                    Numeric.Float0)
  
#
#  Set resources for titling.
#
resources = Resources()
resources.tiXAxisString = "X"
resources.tiYAxisString = "Y"
  
plot = ngl_xy(wks,x,y2,resources)    # Draw the plot.
  
#----------- Begin third plot -----------------------------------------
  
resources.xyLineColors        = [2,3,4]    # Define line colors.
resources.xyLineThicknesses   = [1.,2.,5.] # Define line thicknesses
                                           # (1.0 is the default).
  
plot = ngl_xy(wks,x,y2,resources)         
  
#---------- Begin fourth plot ------------------------------------------
  
resources.tiMainString    = "X-Y plot"  # Title for the XY plot
resources.tiXAxisString   = "X Axis"    # Label for the X axis
resources.tiYAxisString   = "Y Axis"    # Label for the Y axis
resources.tiMainFont      = "Helvetica" # Font for title
resources.tiXAxisFont     = "Helvetica" # Font for X axis label
resources.tiYAxisFont     = "Helvetica" # Font for Y axis label
  
resources.xyMarkLineModes = ["Lines","Markers","MarkLines"]
resources.xyMarkers       = [0,1,3]     # (none, dot, asterisk)
resources.xyMarkerColor   = 3           # Marker color
resources.xyMarkerSizeF   = 0.03        # Marker size (default
                                          # is 0.01)
  
plot = ngl_xy(wks,x,y2,resources)       # Draw an XY plot.
  
#---------- Begin fifth plot ------------------------------------------
  
filename = ncargpath("data") + "/asc/xy.asc"
data = ngl_asciiread(filename,(129,4),"float")
  
#
#  Define a two-dimensional array of data values based on
#  columns two and three of the input data.
#
uv = Numeric.zeros((2,129),Numeric.Float0)
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
resources = Resources()

resources.tiMainString           = "U/V components of wind"
resources.tiXAxisString          = "longitude"
resources.tiYAxisString          = "m/s"
resources.tiXAxisFontHeightF     = 0.02        # Change the font size.
resources.tiYAxisFontHeightF     = 0.02
  
resources.xyLineColors           = [3,4]       # Set the line colors.
resources.xyLineThicknessF       = 2.0         # Double the width.
  
resources.xyLabelMode            = "Custom"    # Label XY curves.
resources.xyExplicitLabels       = ["U","V"]   # Labels for curves
resources.xyLineLabelFontHeightF = 0.02        # Font size and color
resources.xyLineLabelFontColor   = 2           # for line labels
  
plot = ngl_xy(wks,lon,uv,resources) # Draw an XY plot with 2 curves.

# Clean up.
del plot 
del resources

ngl_end()
