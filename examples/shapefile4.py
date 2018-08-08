#
#  File:
#    shapefile4.py
#
#  Synopsis:
#    Drawing France administrative areas using data on a shapefile.
#
#  Categories:
#    Maps only
#    Polylines
#    Polygons
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    April 2014
#
#  Description:
#
#  Effects illustrated:
#      o Plotting data from a shapefile
#      o Zooming in on France on a cylindrical equidistant map
#      o Using named colors
#      o Using gsSegments to specify shapefile segments
# 
#  Output:
#     Two frames showing a polyline plot and a polygon plot.
#
#  Notes:
#     You must download the "FRA_adm" shapefiles from:
#      http://gadm.org/country/
#  
from __future__ import print_function
import numpy,os,sys

#
#  Import Ngl,Nio support functions.
#
import Ngl, Nio

wks_type = "png"
wks = Ngl.open_wks (wks_type,"shapefile4")

dir      = "FRA_adm/"
filename = "FRA_adm1.shp"
if(not os.path.exists(os.path.join(dir,filename))):
  print("You do not have the necessary shapefile file(s) ({}) to run this example.".format(filename))
  print("The comments at the top of this script tell you how to get the files.")
  sys.exit()

f = Nio.open_file(os.path.join(dir,filename), "r")   # Open shapefile

#
# Read data off shapefile
#
lon      = numpy.ravel(f.variables["x"][:])
lat      = numpy.ravel(f.variables["y"][:])
segments = f.variables["segments"]

# First frame: polylines

res = Ngl.Resources()

res.nglFrame            = False
res.nglDraw             = False

# Zoom in on France.
res.mpLimitMode         = "LatLon"
res.mpMinLatF           = 41
res.mpMaxLatF           = 51.1
res.mpMinLonF           = -5.15
res.mpMaxLonF           = 9.6

res.mpFillOn            = False
res.mpOutlineOn         = False

res.tiMainString        = filename
res.tiMainFontHeightF   = 0.015   # Make font slightly smaller.

plot = Ngl.map(wks,res)   # Just create map, don't draw it.

#*************************************************
# Section to add polylines to map.
#*************************************************

plres                  = Ngl.Resources()       # resources for polylines
plres.gsLineColor      = "navyblue"
plres.gsLineThicknessF = 2.0                  # default is 1.0
plres.gsSegments       = segments[:,0]

# This can be slow. Need to investigate.
lnid = Ngl.add_polyline(wks, plot, lon, lat, plres)

Ngl.draw(plot)       # This draws map and attached polylines
Ngl.frame(wks)       # Advance frame.

# Second frame: filled polygons
plot = Ngl.map(wks,res)   # Just create map, don't draw it.

colors = ["antiquewhite3","brown3","navyblue","orange",\
          "cadetblue2","tan1","forestgreen","royalblue",\
          "darkslategray4","sandybrown","plum3","lemonchiffon",\
          "palegreen","khaki3","slateblue4","yellow","violetred",\
          "wheat1","purple","mediumspringgreen","peachpuff2",\
          "orchid4"]

plres.gsColors = colors
gnid = Ngl.add_polygon(wks, plot, lon, lat, plres)

Ngl.draw(plot)       # This draws map and attached polygons
Ngl.frame(wks)       # Advance frame.

Ngl.end()


