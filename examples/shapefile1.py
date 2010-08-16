#
#  File:
#    shapefile1.py
#
#  Synopsis:
#    Illustrates reading data from a shapefile and coloring U.S. states
#    by "Percent unemployment".
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
#    This example reads shapefile data from  the National Atlas
#    (http://www.nationalatlas.gov/) and color fills the states 
#    based upon "percent unemployment", which is calculated from
#    several of the non-spatial variables in the file.
#
#    You must also have the files "states.dbf" and "states.shx"
#    for this example to run. You can look for the data files at:
#
#    http://www.ncl.ucar.edu/Applications/Data/#shp
#
#  Effects illustrated:
#      o Plotting data from a shapefile
#      o Drawing a custom labelbar on a map
#      o Drawing filled polygons over a Lambert Conformal plot
#      o Zooming in on a particular area on a Lambert Conformal map
# 
#  Output:
#     A single visualization is produced showing the filled U.S. states.
#

#
#  Import numpy and os
#
import numpy,os

#
#  Import Ngl,Nio support functions.
#
import Ngl, Nio

wks_type = "ps"
wks = Ngl.open_wks (wks_type,"shapefile1")
Ngl.define_colormap(wks,"rainbow+gray")

#
#  Map resources.
#
res = Ngl.Resources()

res.mpProjection        = "LambertConformal"

res.mpLambertParallel1F = 33         # two parallels
res.mpLambertParallel2F = 45
res.mpLambertMeridianF  = -98        # central meridian

res.mpLimitMode         = "Corners"   # limit map via two opposite corners
res.mpLeftCornerLatF    = 22          # left corner
res.mpLeftCornerLonF    = -125        # left corner
res.mpRightCornerLatF   = 50          # right corner
res.mpRightCornerLonF   = -64         # right corner

res.mpFillOn               = True            # Turn on fill for map areas.
res.mpLandFillColor        = "LightGray"
res.mpOceanFillColor       = "Cyan"
res.mpInlandWaterFillColor = "Cyan"

res.pmTickMarkDisplayMode = "Always"        # Turn on map tickmarks

res.tiMainString          = "Percentage unemployment, by state"

res.nglDraw  = False       # don't draw the plots now
res.nglFrame = False       # or advance the frame

plot = Ngl.map(wks,res) # create the map plot

#
# Read data off shapefile. Must have states.shp, states.dbf,
# and states.prj file in this directory.
#
dirc = Ngl.pynglpath("data")
f    = Nio.open_file(os.path.join(dirc,"shp","states.shp"), "r")   # Open shapefile
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

unemp = f.variables["UNEMPLOY"][:] / f.variables["PERSONS"][:]
lon   = f.variables["x"][:]
lat   = f.variables["y"][:]

#*************************************************
# Section to add filled polygons to map.
#*************************************************

plres             = Ngl.Resources()      # resources for polylines
plres.gsEdgesOn   = True                 # draw border around polygons
plres.gsEdgeColor = "black"    

colors = ["blue","green","yellow","red"]

segNum = 0

for i in range(0,numFeatures):
# color assignment (probably a better way to do this?)
  if (unemp[i] >= 0.01 and unemp[i] < 0.02):
     plres.gsFillColor = colors[0]

  if (unemp[i] >= 0.02 and unemp[i] < 0.03):
     plres.gsFillColor = colors[1]

  if (unemp[i] >= 0.03 and unemp[i] < 0.04):
     plres.gsFillColor = colors[2]

  if (unemp[i] >= 0.04):
     plres.gsFillColor = colors[3]

  startSegment = geometry[i][geom_segIndex]
  numSegments  = geometry[i][geom_numSegs]
  lines = []
  for seg in range(startSegment, startSegment+numSegments):
    startPT = segments[seg, segs_xyzIndex]
    endPT = startPT + segments[seg, segs_numPnts] - 1
    lines.append(Ngl.add_polygon(wks, plot, lon[startPT:endPT],  \
                                 lat[startPT:endPT], plres))
    segNum = segNum + 1

Ngl.draw(plot)

# Make a labelbar...
labels = [ "1", "2", "3", "4" ]

lres                    = Ngl.Resources()
lres.vpWidthF           = 0.50             # location
lres.vpHeightF          = 0.05             # " " 
lres.lbPerimOn          = False            # Turn off perimeter.
lres.lbOrientation      = "Horizontal"     # Default is vertical.
lres.lbLabelAlignment   = "BoxCenters"     # Default is "BoxCenters".
lres.lbFillColors       = colors
lres.lbMonoFillPattern  = True             # Fill them all solid.
lres.lbLabelFontHeightF = 0.012            # label font height
lres.lbTitleString      = "percent"        # title
lres.lbTitlePosition    = "Bottom"         # location of title
lres.lbTitleFontHeightF = 0.01             # title font height
Ngl.labelbar_ndc (wks,4,labels,0.23,0.15,lres)  
  
Ngl.frame(wks)  # Advance the frame.

Ngl.end()
