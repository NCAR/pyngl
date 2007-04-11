#
#  File:
#    natgrid1.py
#
#  Synopsis:
#    Simple example showing natgrid, nnsetp, and nngetp usage.
#
#  Category:
#    Processing
#    Contouring
#    Label bar
#
#  Author:
#    Fred Clare
#  
#  Date of initial publication:
#    December, 2005
#
#  Description:
#    This example reads data from a NetCDF file, does interpolations
#    from randomply-spaced data to gridded data, then draws a colored
#    contour plots, one plot that allows negative interpolated values
#    and another that does not allow negative interpolated values.
#
#  Effects illustrated:
#    o  Reading from a NetCDF file.
#    o  Interpolating randomly-spaced data to gridded data.
#    o  Setting various legend resources.
#    o  Setting natgrid control parameters.
# 
#  Output:
#    This example produces two visualizations:
#      1.) A colored contour plot with a label bar where negative
#          interpolated values are allowed.
#      2.) A colored contour plot with a label bar where negative
#          interpolated values are not allowed.
#
#  Notes:
#     
import numpy
import Ngl

#  
#  Open the ASCII file of seismic data.
#  
dirc    = Ngl.pynglpath("data")
seismic = Ngl.asciiread(dirc + "/asc/seismic.asc" ,[52,3],"float")

#
#  Extract coordinate triples from the data file.
#
x = numpy.array(seismic[:,0],'f')
y = numpy.array(seismic[:,1],'f')
z = numpy.array(seismic[:,2],'f')-785.

#
#  Define output grid for the calls to "natgrid".
#
numxout, numyout = 20, 20 
xmin, xmax, ymin, ymax   = min(x), max(x), min(y), max(y)
xc      = (xmax-xmin)/(numxout-1)
yc      = (ymax-ymin)/(numyout-1)
xo = xmin + xc*numpy.arange(0,numxout)
yo = ymin + yc*numpy.arange(0,numxout)

zo = Ngl.natgrid(x, y, z, xo, yo)   # Interpolate, allow negative values.

#
#  Define a color map and open four different types of workstations.
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
wks = Ngl.open_wks(wks_type,"natgrid1",rlist)

resources                       = Ngl.Resources()
resources.sfXArray              = xo            # X axes data points
resources.sfYArray              = yo            # Y axes data points

resources.tiMainString          = "Negative interpolants allowed"
resources.tiMainFont            = "Times-Roman"
resources.tiMainFontHeightF     = 0.027
resources.tiXAxisString         = "x values"    # X axis label.
resources.tiXAxisFont           = "Helvetica"   
resources.tiXAxisFontHeightF    = 0.02
resources.tiYAxisString         = "y values"    # Y axis label.
resources.tiYAxisFont           = "Helvetica"   
resources.tiYAxisFontHeightF    = 0.02

resources.cnFillOn              = True     # Turn on contour fill.
resources.cnInfoLabelOn         = False    # Turn off info label.
resources.cnLineLabelsOn        = False    # Turn off line labels.

resources.lbOrientation         = "Horizontal" # Draw it horizontally.
resources.lbLabelFont           = "Helvetica"
resources.lbLabelFontHeightF    = 0.015
resources.lbBottomMarginF       =  0.4  #  Move the label bar up.

resources.tmXBLabelFont         = "Helvetica"
resources.tmXBLabelFontHeightF  = 0.02
resources.tmYLLabelFont         = "Helvetica"
resources.tmYLLabelFontHeightF  = 0.02
                                                 # label bar.
resources.nglSpreadColors = False    # Do not interpolate color space.
resources.vpYF = 0.9                 # Change Y location of plot.

#
#  Contour
#
zt = numpy.transpose(zo)
contour = Ngl.contour(wks,zt,resources) 

#
#  Repeat the above, except disallow negative interpolated values.
#
Ngl.nnsetp("non",1)
resources.tiMainString          = "Negative interpolants not allowed"
zo = Ngl.natgrid(x, y, z, xo, yo)
zt = numpy.transpose(zo)
contour = Ngl.contour(wks,zt,resources) 

#
#  Retrieve the default value for the tautness factor "bJ".
#
bJ_value = Ngl.nngetp("bJ")
# print bJ_value

Ngl.end()
