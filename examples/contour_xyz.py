#
#  File:
#    contour_xyz.py
#
#  Synopsis:
#    Illustrates contouring randomly-spaced data.
#
#  Categories:
#    Contouring on non-rectangular grids and raster contouring.
#    Processing
#    Polymarkers
#
#  Author:
#    Dave Brown
#  
#  Date of initial publication:
#    October, 2005
#
#  Description:
#    This example shows the various ways that you can take 
#    three one-dimensional arrays of the same length 
#    (X,Y coordinate points with a corresponding data value) 
#    and generate a contour plot. The first method used is 
#    to interpolate the data to a 2D grid and then contour
#    the 2D grid. The second method is to generate the contours 
#    using triangulation. With both methods, both "area fill" 
#    and "raster fill" methods are used, to show the differences.
#
#  Effects illustrated:
#    o  Using Natgrid to interpolate 2D random data to a 
#       rectangular grid.
#    o  Raster contouring.
#    o  Area fill contouring.
#    o  Triangulation of the input data.
# 
#  Output:
#    This example produces four visualizations:
#      1.)  Contouring randomly-spaced data via interpolation
#           to a regualar grid using area fill with the original
#           data positions marked.
#      2.)  Same as 1.) but using raster fill.
#      3.)  Triangulating the data and contouring using raster fill
#           on the tringular mesh.
#      4.)  Same as 3.) but using area fill.
#
#  Notes:
#     

# 
# Contouring random data
#
#
#  Import numpy.
#
import numpy

#
#  Import Ngl support functions.
#
import Ngl
import Nio

#  
#  Open the ASCII file.
#  
dirc    = Ngl.pynglpath("data")
seismic = Ngl.asciiread(dirc + "/asc/seismic.asc" ,[52,3],"float")

#
#  Read in X,Y,Z data.
#
x = numpy.array(seismic[:,0],'f')
y = numpy.array(seismic[:,1],'f')
z = numpy.array(seismic[:,2],'f')

#
#  Interpolate to a 2D output grid using natgrid.
#
numxout = 20     # Define output grid for call to "natgrid".
numyout = 20
xmin    = min(x)
ymin    = min(y)
xmax    = max(x)
ymax    = max(y)

xc      = (xmax-xmin)/(numxout-1)
yc      = (ymax-ymin)/(numyout-1)

xo = xmin + xc*numpy.arange(0,numxout)
yo = ymin + yc*numpy.arange(0,numxout)
zo = Ngl.natgrid(x, y, z, xo, yo)   # Interpolate.

#
#  Define a color map.
#
cmap = numpy.array([[1.00, 1.00, 1.00], [0.00, 0.00, 0.00], \
                      [1.00, 0.00, 0.00], [1.00, 0.00, 0.40], \
                      [1.00, 0.00, 0.80], [1.00, 0.20, 1.00], \
                      [1.00, 0.60, 1.00], [0.60, 0.80, 1.00], \
                      [0.20, 0.80, 1.00], [0.20, 0.80, 0.60], \
                      [0.20, 0.80, 0.00], [0.20, 0.40, 0.00], \
                      [0.20, 0.45, 0.40], [0.20, 0.40, 0.80], \
                      [0.60, 0.40, 0.80], [0.60, 0.80, 0.80], \
                      [0.60, 0.80, 0.40], [1.00, 0.60, 0.80]],'f')

rlist = Ngl.Resources()
rlist.wkColorMap = cmap
wks_type = "ps"
wks   = Ngl.open_wks( wks_type,"contour_xyz",rlist)  # Open a PS workstation 
                                                     # with the given colormap.

#
# Set up a resource list to contain the resources for our various
# contour plots.
#
resources                       = Ngl.Resources()
resources.nglFrame              = False         # Don't advance frame.
resources.sfXArray              = xo            # X axis data points
resources.sfYArray              = yo            # Y axis data points

resources.tiMainString          = "Interpolation to a 2D grid (AreaFill)"
resources.tiMainFont            = "Helvetica-Bold"
resources.tiXAxisString         = "x values"    # X axis label.
resources.tiYAxisString         = "y values"    # Y axis label.

resources.cnFillOn              = True     # Turn on contour fill.
resources.cnInfoLabelOn         = False    # Turn off info label.
resources.cnLineLabelsOn        = False    # Turn off line labels.

resources.lbOrientation         = "Horizontal"   # Draw labelbar horizontally.

resources.nglSpreadColors = False         # Do not interpolate color space.
resources.vpYF            = 0.9           # Change Y location of plot.

#
#  Transpose the array so we can plot it, because Natgrid returns it
#  as an X x Y array, rather than Y x X which PyNGL expects.
#
zt = numpy.transpose(zo)

contour = Ngl.contour(wks,zt,resources)

#
# Add some polymarkers showing the original locations of the X,Y points.
#
poly_res               = Ngl.Resources()
poly_res.gsMarkerIndex = 16
poly_res.gsMarkerSizeF = 0.015
Ngl.add_polymarker(wks,contour,x,y,poly_res)

Ngl.draw(contour)
Ngl.frame(wks)


#
# Now turn on RasterFill for the contours and compare.
#
resources.tiMainString          = "Interpolation to a 2D grid (RasterFill)"
resources.cnFillMode            = 'RasterFill'
contour                         = Ngl.contour(wks,zt,resources)

#
#  Draw polymarkers again.
#
Ngl.add_polymarker(wks,contour,x,y,poly_res)
Ngl.draw(contour)
Ngl.frame(wks)


#
#  Now draw contours using the triangulation method. Here, we use the
#  X,Y,Z points directly.  The automatic triangulation of the data is 
#  triggered by "sfXArray", "sfYarray", and "z" being 1D arrays of the 
#  same size.
#
resources.sfXArray              = x            # X axis data points
resources.sfYArray              = y            # Y axis data points
resources.cnRasterCellSizeF     = 0.005
resources.tiMainString          = "Delauney triangulation (RasterFill)"

contour = Ngl.contour(wks,z,resources)

#
#  Draw the polymarkers.
#
Ngl.add_polymarker(wks,contour,x,y,poly_res)
Ngl.draw(contour)
Ngl.frame(wks)

#
#  Now use AreaFill instead of RasterFill to compare.
#
resources.cnFillMode   = 'AreaFill'
resources.tiMainString = "Delauney triangulation (AreaFill)"
contour = Ngl.contour(wks,z,resources)

#
#  Draw the polymarkers.
#
Ngl.add_polymarker(wks,contour,x,y,poly_res)
Ngl.draw(contour)
Ngl.frame(wks)

del contour

Ngl.end()
