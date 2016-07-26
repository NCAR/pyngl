#
#  File:
#    newcolor4.py
#
#  Synopsis:
#    Illustrates new color capabilities in PyNGL 1.5.0
#
#  Categories:
#    xy plots
#
#  Author:
#    Mary Haley, based on NCL example
#  
#  Date of initial publication:
#    November 2012
#
#  Description:
#    This example shows how to draw nine plots with three different colormaps
#
#  Effects illustrated:
#    o  Using the new "cnFillPalette" resource.
#    o  Panelling plots.
#    o  Drawing an Aitoff map
#    o  Generating dummy data
# 
#  Output:
#     A single visualization with nine plots
#
import numpy,os
import Ngl,Nio

#---Start the graphics section
wks_type = "png"
wks = Ngl.open_wks (wks_type,"newcolor4")

#---Generate some dummy lat/lon data
nlat      =  64
nlon      = 128
lat       = Ngl.fspan(-90,90,nlat)
lon       = Ngl.fspan(-178.5,178.5,nlon)

#---Values to use for contour labelbar
dmins = [-20.,-10.,-21.]     # Data mins
dmaxs = [ 16., 10., 17.]     # Data maxs
dspas = [  4.,  1.,  3.]     # Data spacing

# One color map per row of plots
colormaps = ["wgne15","StepSeq25","BlueDarkRed18"]

# Create resource list for customizing contour over maps
res                        = Ngl.Resources()

res.nglMaximize            = False
res.nglFrame               = False
res.nglDraw                = False

res.mpProjection           = "Aitoff"
res.mpOutlineOn            = True

res.mpPerimOn              = False
res.mpGridAndLimbOn        = False
res.pmTickMarkDisplayMode  = "Never"

res.cnFillOn               = True
res.cnLinesOn              = False
res.cnLineLabelsOn         = False
res.cnLevelSelectionMode   = "ManualLevels"
res.lbLabelBarOn           = False

res.sfXArray               = lon
res.sfYArray               = lat

#
# Loop 9 times and create 9 dummy plots; each group
# of 3 plots has the same color map.
#
nplots = 9
plots  = []
for n in range(0,nplots):
  print("plot #",n)
  dmin = dmins[n/3]
  dmax = dmaxs[n/3]
  dspa = dspas[n/3]

  mstart = numpy.random.uniform(10,25,1).astype(int)
  mend   = numpy.random.uniform(10,25,1).astype(int)
  xstart = numpy.random.uniform(dmin,dmin+2,1)
  xend   = numpy.random.uniform(dmax-2,dmax,1)

#---This is a new resource added in PyNGL 1.5.0
  res.cnFillPalette          = colormaps[n/3]

  res.cnMinLevelValF         = dmin
  res.cnMaxLevelValF         = dmax
  res.cnLevelSpacingF        = dspa

  data = Ngl.generate_2d_array([nlat,nlon],mstart,mend,xstart,xend)

  plots.append(Ngl.contour_map(wks,data,res))

# Resources for panelling
pres                  = Ngl.Resources() 
pres.nglFrame         = False
pres.nglPanelLabelBar = True

# Calculate start Y position for first row of plots
height = 0.15            # we know this will be height of small plots
extra  = 1.0-(3*height)
top    = 1.0-(extra/2.)

# Draw a title before we draw plots
title               = "Multiple panels on one page, 3 different colormaps"
txres               = Ngl.Resources()
txres.txJust        = "BottomCenter"
txres.txFontHeightF = 0.02
Ngl.text_ndc(wks,title,0.5,top+0.01,txres)

# Loop across plots and panel them on one page
for n in range(0,3):
# Define location in a unit square for each set of plots.
  pres.nglPanelTop    = top-(n*height)
  pres.nglPanelBottom = top-((n+1)*height)

  Ngl.panel(wks,plots[n*3:n*3+3],[1,3],pres)
  
Ngl.frame(wks)
Ngl.end()
