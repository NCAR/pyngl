#
#  File:
#    streamline2.py
#
#  Synopsis:
#    Draws streamlines colored by a scalar field.over contours and a map.
#
#  Category:
#    Streamlines
#
#  Author:
#    Mary Haley (based on NCL/HLU example)
#  
#  Date of initial publication:
#    September, 2007
#
#  Description:
#    This example draws streamlines overlaid on contours on a map,
#    and then streamlines colored by a scalar field on a map.
#
#  Effects illustrated:
#    o  Selecting a color map by name.
#    o  Adding a color to the color map.
#    o  Overlaying multiple plots on a single plot.
#    o  Copying one resource list to another.
#    o  Drawing colored streamlines.
#    o  Maximizing a plot in the frame.
# 
#  Output:
#    This example produces two frames: one with streamlines and
#    contours over a map, and the second with streamlines colored
#    by a scalar field.
#
#  Notes:
#     

#
#  Import numpy.
#
import numpy

#
#  Import Nio for a NetCDF reader.
#
import Nio

#
#  Import PyNGL support functions.
#
import Ngl
import os,string,types

# Read data off file.

filename = os.path.join(Ngl.pynglpath("data"),"asc","fcover.dat")
data     = Ngl.asciiread(filename,[3,73,73],"float")

# Open workstation and change color map.
wks_type = "ps"

rlist = Ngl.Resources()
rlist.wkColorMap = "so4_23"
if(wks_type == "ps" or wks_type == "pdf"):
  rlist.wkOrientation = "Portrait"      # For PS or PDF output only.

wks = Ngl.open_wks(wks_type,"streamline2",rlist)
gray  = Ngl.new_color(wks,0.88,0.88,0.88)      # add gray to colormap

stres = Ngl.Resources()
cnres = Ngl.Resources()
mpres = Ngl.Resources()

cnres.nglDraw  = False
cnres.nglFrame = False
mpres.nglDraw  = False
mpres.nglFrame = False
stres.nglDraw  = False
stres.nglFrame = False

# Set sf/vf resources to indicate where on map to overlay.
cnres.sfXCStartV             = -180.
cnres.sfXCEndV               =  180.
cnres.sfYCStartV             =  -90.
cnres.sfYCEndV               =   90.
stres.vfXCStartV             = -180.
stres.vfXCEndV               =  180.
stres.vfYCStartV             =  -90.
stres.vfYCEndV               =   90.

stres.stLineColor            = 8      # green
cnres.cnLineColor            = 6      # blue
stres.stLineThicknessF       = 1.5
cnres.cnLineThicknessF       = 1.7
cnres.cnLineDashPattern      = 7
cnres.cnLineLabelsOn         = False

# Set map resources.
mpres.mpGridAndLimbOn        = False
mpres.mpCenterLatF           = 90.0
mpres.mpCenterLonF           = 180.0
mpres.mpCenterRotF           = 45.0
mpres.mpFillOn               = True
mpres.mpGridAndLimbDrawOrder = "Draw"
mpres.mpGridLineDashPattern  = 5
mpres.mpInlandWaterFillColor = -1
mpres.mpOceanFillColor       = -1
mpres.mpLandFillColor        = gray
mpres.mpLabelsOn             = False
mpres.mpLeftCornerLatF       = 10.
mpres.mpLeftCornerLonF       = -180.
mpres.mpLimitMode            = "corners"
mpres.mpProjection           = "Stereographic"
mpres.mpRightCornerLatF      = 10.
mpres.mpRightCornerLonF      = 0.

mpres.tiMainFontHeightF      = 0.02
mpres.tiMainString           = "Busy graphic with contours and streamlines"

stream  = Ngl.streamline(wks,data[0,:,:],data[1,:,:],stres)
contour = Ngl.contour(wks,data[2,:,:],cnres)
map     = Ngl.map(wks,mpres)

Ngl.overlay(map,stream)
Ngl.overlay(map,contour)

# Since we overlaid 2 plots, we need to resize them to make sure
# they fit in frame.
Ngl.maximize_plot(wks,map)

Ngl.draw(map)
Ngl.frame(wks)

del cnres.nglDraw
del cnres.nglFrame
del mpres.nglDraw
del mpres.nglFrame
del stres.nglDraw
del stres.nglFrame
#
# Copy all three resource lists to one big resource list.
#
resources = Ngl.Resources()
for t in dir(cnres):
  if len(t) > 5 and t[0:6] != 'cnLine':
    setattr(resources,t,getattr(cnres,t))

for t in dir(mpres):
  setattr(resources,t,getattr(mpres,t))

for t in dir(stres):
  setattr(resources,t,getattr(stres,t))

#resources.pmLabelBarDisplayMode = "Always"
resources.nglSpreadColorStart   = 3
resources.nglSpreadColorEnd     = -4
resources.stMonoLineColor       = False
resources.stLineThicknessF      = 1.7
resources.tiMainString          = "Streamlines colored by scalar field"

stream = Ngl.streamline_scalar_map(wks,data[0,:,:],data[1,:,:],data[2,:,:], \
                                   resources)
Ngl.end()
