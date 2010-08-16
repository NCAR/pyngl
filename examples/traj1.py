#
#  File:
#    traj1.py
#
#  Synopsis:
#    Draws lines and polymarkers over a map to trace the paths of bouys.
#
#  Categories:
#    Maps
#    Polylines
#    Polygons
#    Polymarkers
#
#  Author:
#    Fred Clare
#  
#  Date of initial publication:
#    September, 2004
#
#  Description:
#    Draws lines and polymarkers over a map to trace the paths of 
#    buoys set afloat off the eastern coast of South America.
#    The colors of path segments indicate salinity measurements.
#
#  Effects illustrated:
#    o  Reading from a NetCDF file.
#    o  Polylines.
#    o  Polygons.
#    o  Polymarkers.
#    o  Maps using a Cylindrical Equidistant projection.
# 
#  Output:
#    A single visualization showing the paths of buoys set afloat
#    off the eastern coast of South America.
#
#  Notes:
#    This program implements a version of a trajectory plot
#    based on an NCL code developed by NCAR's Climate and
#    Global Dynamics Division.
#

#
#  Import numpy.
#
import numpy, os

#
#  Import Nio.
#
import Nio

#
#  Import Ngl support functions.
#
import Ngl

#
#  nint returns the nearest integer to a given floating value.
#
#
def nint(r):
  if (abs(int(r)-r) < 0.5):
    return int(r)
  else:
    if (r >= 0.):
      return int(r)+1
    else:
      return int(r)-1

#
#  Open the netCDF file containing the salinity data for the trajectories.
#
dirc = Ngl.pynglpath("data")
ncdf = Nio.open_file(os.path.join(dirc,"cdf","traj_data.nc"),"r")

#
#  Define a color map and open a workstation.
#
cmap = numpy.array([[1.00, 1.00, 1.00], [0.00, 0.00, 0.00], \
                    [0.88, 0.57, 0.20], [0.90, 0.90, 0.95], \
                    [1.00, 0.00, 0.00], [0.00, 1.00, 0.00], \
                    [0.00, 0.00, 1.00], [0.00, 1.00, 1.00], \
                    [1.00, 0.00, 1.00], [1.00, 1.00, 0.00], \
                    [0.80, 0.50, 0.50], [0.20, 0.80, 0.20], \
                    [0.80, 0.80, 1.00], [0.00, 0.00, 0.00]],\
                    'f')

rlist = Ngl.Resources()
rlist.wkColorMap = cmap
wks_type = "ps"
wks = Ngl.open_wks (wks_type,"traj1",rlist)

#
#  Create the plot.
#

#
#  Map resources.
#
res            = Ngl.Resources()   # map resources
res.nglFrame   = False         # don't advance frame
res.vpWidthF   = 0.80          # make map bigger
res.vpHeightF  = 0.80
res.mpLimitMode = "LatLon"
res.mpMaxLatF  = -20           # select subregion
res.mpMinLatF  = -60
res.mpMinLonF  = -75
res.mpMaxLonF  = -25
res.mpFillOn   = True
res.mpLabelsOn = True
res.mpOutlineOn = True
res.mpLandFillColor = 2
res.mpOceanFillColor = 3
res.mpGridAndLimbOn = False
res.vpXF       = 0.05
res.vpYF       = 0.95
res.vpWidthF   = 0.8
res.vpHeightF  = 0.8
res.tmXBTickStartF = -75
res.tmXBTickEndF = -25
res.tmYROn  = True
res.tmXTOn  = True
res.tmXBLabelFontHeightF = 0.02
res.tmYLLabelFontHeightF = 0.02
res.mpProjection = "CylindricalEquidistant"
map = Ngl.map(wks,res)    # Draw map.

#
#  Main title.
#
txres = Ngl.Resources()
txres.txFontHeightF = 0.025
txres.txFontColor   =  1
txres.txFont        = 22
Ngl.text_ndc(wks,"Trajectories colored by salinity (ppt)",0.52,0.90,txres)
del txres

#
#  Manually draw a labelbar using colors 4-12.
#
gsres = Ngl.Resources()  # Line resources.
gsres.gsLineColor = 1    # Draw black boundaries around the label bar boxes.
delx = 0.06
dely = 0.03
startx = 0.30
starty = 0.25
txres = Ngl.Resources()          # For labelling the label bar.
txres.txFontHeightF = 0.018 
txres.txFontColor   = 1
txres.txFont   = 22
for i in xrange(4,14,1):
  x0 = startx+(i-4)*delx
  x1 = x0+delx
  x2 = x1
  x3 = x0
  x4 = x0
  x = [x0,x1,x2,x3,x4]
  y0 = starty
  y1 = y0
  y2 = y1+dely
  y3 = y2
  y4 = y0
  y = [y0,y1,y2,y3,y4]
  gsres.gsFillColor = i    # Change fill color.
  Ngl.polygon_ndc(wks,x,y,gsres) 
  Ngl.polyline_ndc(wks,x,y,gsres)
  if (i == 4):
    Ngl.text_ndc(wks,"34.55",x0,y0-dely,txres)
  elif(i == 6):
    Ngl.text_ndc(wks,"34.61",x0,y0-dely,txres)
  elif(i == 8):
    Ngl.text_ndc(wks,"34.67",x0,y0-dely,txres)
  elif(i == 10):
    Ngl.text_ndc(wks,"34.73",x0,y0-dely,txres)
  elif(i == 12):
    Ngl.text_ndc(wks,"34.79",x0,y0-dely,txres)
  Ngl.text_ndc(wks,"34.85",startx+10*delx,y0-dely,txres)

#
#  Draw the trajectories.
#
traj = [1,10,53,67,80]   # choose which trajectories to plot
   
pres                  = Ngl.Resources()        # polyline resources
pres.gsLineThicknessF = 3.0                # line thickness
mres  = Ngl.Resources()                        # marker resources
mres.gsMarkerSizeF  = 17.0        # marker size
mres.gsMarkerColor  = "black"     # marker color


#
#  Loop through the chosen trajectories.
#
sdata = ncdf.variables["sdata"]
for i in xrange(len(traj)):
  ypt = sdata[2,:,traj[i]]                  # extract lat from whole array
  xpt = sdata[1,:,traj[i]]                  # extract lon from whole array
  sst = sdata[8,:,traj[i]]

#
#  Map the salinity values between 34.55 and 34.85 into color
#  indices between 4 and 13 and draw polylines, with those 
#  colors, between adjacent lat/lon values.
#
  for j in xrange(len(ypt)-2):
    sval = 0.5*(sst[j]+sst[j+1])
    cindex = nint(4.+30.*(sval-34.55))
    if (cindex < 4):
      cindex = 4
    elif (cindex > 13):
      cindex = 13
    pres.gsLineColor = cindex
    Ngl.polyline(wks,map,[xpt[j],xpt[j+1]],[ypt[j],ypt[j+1]],pres)

#
#  Draw a polymarker at the beginning of each trajectory.
#
    Ngl.polymarker(wks,map,[xpt[0]],[ypt[0]],mres) 

Ngl.frame(wks)

Ngl.end()
