#
#  File:
#    overlay1.py
#
#  Synopsis:
#    Shows how to overlay multiple plots on one plot and then remove them.
#
#  Category:
#    Vectors and contours over maps.
#    Masked arrays
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    December, 2005
#
#  Description:
#    This example draws vectors and contours over maps illustrating
#    how to overlay multiple plots and then remove them.
#
#  Effects illustrated:
#    o  Drawing vectors and contours over specified map regions.
#    o  Overlaying plots
#    o  Removing overlaid plots.
#    o  Maximizing a plot in the frame.
#    o  Using masked arrays to handle missing values.
# 
#  Output:
#    This example produces three visualizations:
#      1.)  Vectors, line and filled contours over the U.S.
#      2.)  Line and filled contours over the U.S.
#      3.)  Filled contours over the U.S.
#
#  Notes:
#     

#
#  Import numpy.
#
import numpy, os

#
#  Import Nio for a NetCDF reader.
#
import Nio

#
#  Import PyNGL support functions.
#
import Ngl

#
#  Import masked array module.
#
from numpy import ma

#
#  Open netCDF files.
#
dirc = Ngl.pynglpath("data")

#
#  Open the netCDF files.
#
ufile = Nio.open_file(os.path.join(dirc,"cdf","Ustorm.cdf"),"r")
vfile = Nio.open_file(os.path.join(dirc,"cdf","Vstorm.cdf"),"r")
tfile = Nio.open_file(os.path.join(dirc,"cdf","Tstorm.cdf"),"r")
pfile = Nio.open_file(os.path.join(dirc,"cdf","Pstorm.cdf"),"r")

#
#  Get the u/v, temperature, and pressure variables.
#
u = ufile.variables["u"]
v = vfile.variables["v"]
t = tfile.variables["t"]
p = pfile.variables["p"]
lat = ufile.variables["lat"]
lon = ufile.variables["lon"]
ua = ma.masked_values(u[0,:,:],u._FillValue)
va = ma.masked_values(v[0,:,:],v._FillValue)
pa = ma.masked_values(p[0,:,:],p._FillValue)
ta = ma.masked_values(t[0,:,:],t._FillValue)

#
# Scale the temperature and pressure data.
#
ta = (ta-273.15)*9.0/5.0+32.0
pa = 0.01*pa

#
# Open a PostScript workstation and change the color map.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"overlay1")

#
# Set up resource lists for vector, line contour, and
# filled contour plots.
#
mpres = Ngl.Resources()
vcres = Ngl.Resources()
clres = Ngl.Resources()
cfres = Ngl.Resources()

#
# Turn off nglDraw and nglFrame because we don't want to draw all
# these plots until they are all overlaid on the map plot.
#
mpres.nglDraw  = False
mpres.nglFrame = False
vcres.nglDraw  = False
vcres.nglFrame = False
clres.nglDraw  = False
clres.nglFrame = False
cfres.nglDraw  = False
cfres.nglFrame = False

#
# Set up coordinates of X and Y axes for all plots. This
# is necessary in order for the Ngl.overlay calls to work
# later.
#
vcres.vfXArray = lon[:]
vcres.vfYArray = lat[:]
clres.sfXArray = lon[:]
clres.sfYArray = lat[:]
cfres.sfXArray = lon[:]
cfres.sfYArray = lat[:]

#
# Set additional resources needed.
#
mpres.pmTitleDisplayMode        = "Always"       # Turn on map title.
mpres.tiMainString              = "map only"
mpres.mpLimitMode               = "LatLon"       # Change the area of the
mpres.mpMinLatF                 =  18.0          # map viewed.
mpres.mpMaxLatF                 =  65.0
mpres.mpMinLonF                 = -129.
mpres.mpMaxLonF                 = -57.
mpres.mpOutlineBoundarySets     = "GeophysicalAndUSStates"
mpres.mpGridAndLimbOn           = False         # Turn off grid and limb lines.

vcres.vcMinFracLengthF          = 0.33          # Increase length of
vcres.vcMinMagnitudeF           = 0.001         # vectors.
vcres.vcRefLengthF              = 0.045
vcres.vcRefMagnitudeF           = 20.0
vcres.vcRefAnnoOn               = False

cfres.cnFillOn                  = True          # Turn on contour fill.
cfres.cnLinesOn                 = False         # Turn off contour lines.
cfres.cnLineLabelsOn            = False         # Turn off contour line labels.
cfres.lbOrientation             = "Horizontal"  # horizontal labelbar
cfres.lbLabelFontHeightF        = 0.012         # Decrease font size.
cfres.pmLabelBarOrthogonalPosF  = -0.05         # Move labelbar up.

clres.cnLineLabelsOn            = False         # Turn off contour line labels.
clres.cnLineDashPatterns        = 3             # dashed contour lines
clres.cnLineThicknessF          = 3.            # triple thick contour lines
clres.cnInfoLabelOrthogonalPosF = -0.15         # Move info label up.

#
# Create the various plots. They will not get drawn because
# nglDraw is set to False for all of them.
#
map_plot          = Ngl.map(wks,mpres)
vector_plot       = Ngl.vector(wks,ua,va,vcres)
line_contour_plot = Ngl.contour(wks,pa,clres)
fill_contour_plot = Ngl.contour(wks,ta,cfres)

#
# Overlay everything on the map plot.
#
Ngl.overlay(map_plot,fill_contour_plot)
Ngl.overlay(map_plot,line_contour_plot)
Ngl.overlay(map_plot,vector_plot)

#
# Change the title.
#
srlist = Ngl.Resources()
srlist.tiMainString = "vectors, line, and filled contours"
Ngl.set_values(map_plot,srlist)

# 
# Draw the map plot, which now contains the vectors and
# filled/line contours.
#
Ngl.maximize_plot(wks,map_plot)    # Maximize size of plot in frame.
Ngl.draw(map_plot)
Ngl.frame(wks)

#
# Change the title.
#
srlist.tiMainString = "line and filled contours"
Ngl.set_values(map_plot,srlist)

#
# Remove the vector plot and redraw. We should now
# just see the line contours and filled contours.
#
Ngl.remove_overlay(map_plot,vector_plot,0)
Ngl.draw(map_plot)
Ngl.frame(wks)

#
# Change the title.
#
srlist.tiMainString = "filled contours"
Ngl.set_values(map_plot,srlist)

#
# Remove the line contour plot and redraw. We 
# should now just see the filled contours.
#
Ngl.remove_overlay(map_plot,line_contour_plot,0)
Ngl.draw(map_plot)
Ngl.frame(wks)

Ngl.end()
