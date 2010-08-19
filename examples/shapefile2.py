#
#  File:
#    shapefile2.py
#
#  Synopsis:
#    Simple example of how to draw selected geometry from a shapefile.
#
#  Categories:
#    Maps only
#    Polygons
#
#  Author:
#    Mary Haley (based on an NCL script of Rick Brownrigg, CISL/NCAR)
#  
#  Date of initial publication:
#    August 2010
#
#  Description:
#
#  Effects illustrated:
#      o Plotting data from a shapefile
#      o Drawing selected data based upon a database query of the shapefile
#      o Decreasing the font size of the main title
#      o Zooming in on South America on a cylindrical equidistant map
#      o Drawing a map using the medium resolution map outlines
#      o Using named colors
# 
#  Output:
#     A single visualization is produced showing the streams in South America.
#
#  Notes:
#     You must download the "HYDRO1k Streams data set for South America
#     as a tar file in shapefile format (2.4 MB) from:
#
# http://eros.usgs.gov/#/Find_Data/Products_and_Data_Available/gtopo30/hydro/samerica
#  
#     Gunzip and untar the file:
#
#       gunzip sa_str.tar.gz
#       tar -xf sa_str.tar
#
#
import numpy,os,sys

#
#  Import Ngl,Nio support functions.
#
import Ngl, Nio

wks_type = "ps"
rlist = Ngl.Resources()
rlist.wkColorMap = ["White","Black","Blue","SlateGray2","NavajoWhite"]
wks = Ngl.open_wks (wks_type,"shapefile2",rlist)

filename = "sa_str.shp"
if(not os.path.exists(filename)):
  print "You do not have the necessary shapefile file(s) to run this example."
  print "The comments at the top of this script tell you how to get the files."
  sys.exit()

f = Nio.open_file(filename, "r")   # Open shapefile

#
# Read data off shapefile
#
segments = f.variables["segments"][:]
geometry = f.variables["geometry"][:]
segsDims = segments.shape
geomDims = geometry.shape

#
# Read global attributes  
#
geom_segIndex = f.geom_segIndex
geom_numSegs  = f.geom_numSegs
segs_xyzIndex = f.segs_xyzIndex
segs_numPnts  = f.segs_numPnts

numFeatures = geomDims[0]


res = Ngl.Resources()

res.nglFrame            = False   # Don't advance frame after plot is
                                  # drawn, b/c we want to draw streams
                                  # first.

res.mpDataBaseVersion   = "MediumRes"    # slightly better resolution

# Zoom in on South America.
res.mpLimitMode         = "LatLon"
res.mpMinLatF           = -60
res.mpMaxLatF           =  15
res.mpMinLonF           = -90
res.mpMaxLonF           = -30

res.mpFillOn            = True
res.mpLandFillColor     = "NavajoWhite"
res.mpOceanFillColor    = "SlateGray2"

res.tiMainString        = "Stream network data for South America"
res.tiMainFontHeightF   = 0.015   # Make font slightly smaller.

plot = Ngl.map(wks,res)   # Draw map, but don't advance frame.

#*************************************************
# Section to add polylines to map.
#*************************************************

plres             = Ngl.Resources()           # resources for polylines
plres.gsLineColor = "blue"

lon    = f.variables["x"]
lat    = f.variables["y"]

for i in range(0,numFeatures):
  startSegment = geometry[i, geom_segIndex]
  numSegments  = geometry[i, geom_numSegs]
  for seg in range(startSegment, startSegment+numSegments):
    startPT = segments[seg, segs_xyzIndex]
    endPT   = startPT + segments[seg, segs_numPnts] - 1
#
# Using Ngl.polyline is *much* faster than Ngl.add_polyline,
# b/c there are hundreds of polylines to add.
#
    Ngl.polyline(wks, plot, lon[startPT:endPT],  \
                            lat[startPT:endPT], plres)

Ngl.frame(wks)       # Advance frame.

Ngl.end()


