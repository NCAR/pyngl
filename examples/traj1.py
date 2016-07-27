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
#    o  Using named colors.
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

#----------------------------------------------------------------------
#  nint returns the nearest integer to a given floating value.
#----------------------------------------------------------------------
def nint(r):
  if (abs(int(r)-r) < 0.5):
    return int(r)
  else:
    if (r >= 0.):
      return int(r)+1
    else:
      return int(r)-1

#----------------------------------------------------------------------
# Create map of southern tip of South America
#----------------------------------------------------------------------
def create_map(wks):
#---Map resources
  res            = Ngl.Resources()   # map resources
  res.nglDraw    = False         # don't draw map
  res.nglFrame   = False         # don't advance frame

  res.mpDataBaseVersion = "MediumRes"   # better map outlines

  res.mpLimitMode = "LatLon"
  res.mpMaxLatF   = -20           # select subregion
  res.mpMinLatF   = -62
  res.mpMinLonF   = -78
  res.mpMaxLonF   = -25

  res.mpOutlineOn     = True
  res.mpGridAndLimbOn = False

  res.mpFillOn               = True
  res.mpLandFillColor        = "beige"
  res.mpOceanFillColor       = "paleturquoise1"
  res.mpInlandWaterFillColor = "paleturquoise1"

  res.tmYROn  = True    # Turn on right tickmarks
  res.tmXTOn  = True    # Turn on top tickmarks

  res.tiMainString = "Trajectories colored by salinity (ppt)"
  res.tiMainFontHeightF = 0.02

  map = Ngl.map(wks,res)    # Draw map.

  return map

#----------------------------------------------------------------------
#  Add the trajectory lines.
#----------------------------------------------------------------------
def add_trajectories(wks,map,filename,cmap):
#---Open the netCDF file containing the salinity data for the trajectories.
  ncfile = Nio.open_file(filename)

  traj = [1,10,53,67,80]   # choose which trajectories to plot
   
  pres                  = Ngl.Resources()    # polyline resources
  pres.gsLineThicknessF = 5.0                # line thickness

  mres  = Ngl.Resources()                    # marker resources
  mres.gsMarkerSizeF       = 17.0            # marker size
  mres.gsMarkerColor       = "black"         # marker color
  mres.gsMarkerIndex       = 15              # circle with an X
  mres.gsMarkerThicknessF  = 2.0             # thicker marker outlines

#
#  Loop through the chosen trajectories.
#
  sdata = ncfile.variables["sdata"]
  sid = []
  for i in xrange(len(traj)):
    lat = sdata[2,:,traj[i]]                  # extract lat from whole array
    lon = sdata[1,:,traj[i]]                  # extract lon from whole array
    sst = sdata[8,:,traj[i]]

#
#  Map the salinity values between 34.55 and 34.85 into color
#  indices between 0 and 9 and draw polylines, with those 
#  colors, between adjacent lat/lon values.
#
    for j in xrange(len(lat)-2):
      sval = 0.5*(sst[j]+sst[j+1])
      cindex = nint(30.*(sval-34.55))
      pres.gsLineColor = cmap[cindex]
      sid.append(Ngl.add_polyline(wks,map,[lon[j],lon[j+1]],[lat[j],lat[j+1]],pres))

#---Draw a polymarker at the beginning of each trajectory.
      sid.append(Ngl.add_polymarker(wks,map,[lon[0]],[lat[0]],mres))

  return

#----------------------------------------------------------------------
# Manually create a labelbar using polygons, lines, and text to 
# draw various primitives.  The lat/lon space of the map is used
# to indicate where to draw the labelbar. By using the various
# Ngl.add_xxxx functions, this actually attaches the primitives to
# the map, so if you were to resize the map later (say in a panel
# plot), the primitives would be resized automatically.
#
# This is the harder way to draw a labelbar, but it shows how to 
# do more custom labelbars, if desired.
#----------------------------------------------------------------------
def add_labelbar(wks,map,cmap):
  gsres = Ngl.Resources()  # Line resources.

  delta_lon = 4.0
  delta_lat = 2.1
  start_lon = -68.
  start_lat = -58.
  txres = Ngl.Resources()          # For labeling the label bar.
  txres.txFontHeightF = 0.015
  gid = []
  lid = []
  tid = []
  for i in xrange(4,14,1):
    lon0 = start_lon+(i-4)*delta_lon
    lon1 = lon0+delta_lon
    lat0 = start_lat
    lat1 = start_lat+delta_lat
    lons = [lon0,lon1,lon1,lon0,lon0]
    lats = [lat0,lat0,lat1,lat1,lat0]
    gsres.gsFillColor = cmap[i-4]    # Change fill color.
    gid.append(Ngl.add_polygon(wks,map,lons,lats,gsres))
    lid.append(Ngl.add_polyline(wks,map,lons,lats,gsres))
    if (i == 4):
      tid.append(Ngl.add_text(wks,map,"34.55",lon0,lat0-delta_lat,txres))
    elif(i == 6):
      tid.append(Ngl.add_text(wks,map,"34.61",lon0,lat0-delta_lat,txres))
    elif(i == 8):
      tid.append(Ngl.add_text(wks,map,"34.67",lon0,lat0-delta_lat,txres))
    elif(i == 10):
      tid.append(Ngl.add_text(wks,map,"34.73",lon0,lat0-delta_lat,txres))
    elif(i == 12):
      tid.append(Ngl.add_text(wks,map,"34.79",lon0,lat0-delta_lat,txres))
    else:
      tid.append(Ngl.add_text(wks,map,"34.85",start_lon+10*delta_lon,lat0-delta_lat,txres))
  
  return

#----------------------------------------------------------------------
#  Main code
#----------------------------------------------------------------------

#---Get the path to the NetCDF file.
dirc     = Ngl.pynglpath("data")
filename = os.path.join(dirc,"cdf","traj_data.nc")

#---Define a list of named colors to use for trajectory lines and labelbar
cmap = ["brown4","yellow1","navyblue","forestgreen","hotpink","purple",\
        "slateblue","thistle","deeppink4","darkgoldenrod"]
         
wks = Ngl.open_wks ("png","traj1")
map = create_map(wks)
add_labelbar(wks,map,cmap)
add_trajectories(wks,map,filename,cmap)

Ngl.draw(map)
Ngl.frame(wks)
Ngl.end()
