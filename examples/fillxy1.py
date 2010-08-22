#
#  File:
#    fillxy1.py
#
#  Synopsis:
#    Illustrates how to fill between curves in an XY plot.
#
#  Categories:
#    xy plots
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    August 2010, based on NCL example
#
#  Description:
#    This example shows how to set a resource for filling between XY curves.
#
#  Effects illustrated:
#    o Filling the area between two curves in an XY plot
#    o Subscripting a variable using coordinate values
# 
#  Output:
#     A single visualization with two XY curves 
#
import numpy,os
import Ngl,Nio

dirc = Ngl.pynglpath("data")
f    = Nio.open_file(os.path.join(dirc,"cdf","uv300.nc"))
u    = f.variables["U"]
lat  = f.variables["lat"][:]

#---To plot multiple lines, you must put them into a multi-dimensional array.
data      = numpy.zeros([2,len(lat)],'f')
data[0,:] = u['time|i0 lat|: lon|82']
data[1,:] = u['time|i0 lat|: lon|-69']

#---Start the graphics section
wks_type = "ps"
wks = Ngl.open_wks (wks_type,"fillxy1")     # Open "fillxy1.ps" PS file.

#---Set some resources.
res                      = Ngl.Resources()

res.tiMainString         = "Filling between two curves"

res.nglXYAboveFillColors = 100    # Indexes into the current color table.
res.nglXYBelowFillColors = 184    # Could also use named colors, e.g. "red"

res.xyLabelMode      = "Custom"          # Custom label the curves
res.xyExplicitLabels = ['Y1','y2']

plot  = Ngl.xy (wks,lat,data,res)        # Draw plot

Ngl.end()

