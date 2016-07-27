#
#  File:
#    mpas3.py
#
#  Synopsis:
#    Draws outlines of an MPAS grid with 349,339 edges
#
#  Categories:
#    contouring
#    polylines
#
#  Author:
#    Mary Haley, based on an NCL example by Dave Brown
#  
#  Date of initial publication:
#    September 2014
#
#  Description:
#    This example shows how to draw a subset of the outline of an MPAS grid
#    over filled contours of temperature.
#
#  Effects illustrated:
#    o  Plotting MPAS data
#    o  Plotting edges of MPAS data on a map
#    o  Using special "gsSegments" for faster polyline draw
#    o  Zooming in on a map
# 
#  Output:
#    This example produces two plots.
#     
#  Notes:
#     The MPAS data file is too large to be included with this 
#     software. Either use your own MPAS grid, or send email to
#     pyngl-talk@ucar.edu.
#
#     http://mailman.ucar.edu/mailman/listinfo/pyngl-talk
#
#======================================================================
import os, numpy, math, time, sys
import Ngl, Nio

#---Code to attach MPAS edge lines to the existing map
def add_mpas_edges(wks,map,mfile):
  lonVertex      = mfile.variables["lonVertex"][:]
  latVertex      = mfile.variables["latVertex"][:]
  latVertex      = latVertex * RAD2DEG
  lonVertex      = lonVertex * RAD2DEG
  verticesOnEdge = mfile.variables["verticesOnEdge"][:]
  nedges         = verticesOnEdge.shape[0]

  ecx = numpy.ndarray((nedges,2),'d')
  ecy = numpy.ndarray((nedges,2),'d')

  ecx[:,0] = lonVertex[verticesOnEdge[:,0]-1]
  ecx[:,1] = lonVertex[verticesOnEdge[:,1]-1]
  ecy[:,0] = latVertex[verticesOnEdge[:,0]-1]
  ecy[:,1] = latVertex[verticesOnEdge[:,1]-1]

  ii   = numpy.where((abs(ecx[:,0]-ecx[:,1]) > 180))
  iigt = numpy.where((ecx[ii,0] > ecx[ii,1]))
  iilt = numpy.where((ecx[ii,0] < ecx[ii,1]))
  ecx[iigt,0] = ecx[iigt,0] - 360.0
  ecx[iilt,1] = ecx[iilt,1] - 360.0

#
# Attach the polylines using special "gsSegments" resource. This
# is *much* faster than attaching every line individually.
#
  lnres                     = Ngl.Resources()
  lnres.gsLineThicknessF    = 0.50             # default is 1
  lnres.gsLineColor         = "Navy"           # default is black.
  lnres.gsSegments          = range(0,nedges * 2,2)
  
  map.poly = Ngl.add_polyline(wks,map,numpy.ravel(ecx),numpy.ravel(ecy),lnres)


mpas_file = "MPASOcean60km.nc"
if(not os.path.exists(mpas_file)):
  print("You do not have the necessary '%s' file to run this example." % mpas_file)
  print("See the comments at the top of this script for more information.")
  sys.exit()

#---Read variable to plot
f              = Nio.open_file(mpas_file)
temp        = f.variables["temperature"][0,:,0]

#---Read edge and lat/lon information
lonCell        = f.variables["lonCell"][:]
latCell        = f.variables["latCell"][:]

#---Convert to degrees from radians
RAD2DEG   = 180.0/(math.atan(1)*4.0)
latCell   = latCell * RAD2DEG
lonCell   = lonCell * RAD2DEG

#---Start the graphics
wks_type = "png"
wks = Ngl.open_wks(wks_type,"mpas3")

res                        = Ngl.Resources()   # Plot mods desired.

res.cnFillOn               = True              # color plot desired
res.cnFillMode             = "RasterFill"
res.cnFillPalette          = "ncl_default"
res.cnLinesOn              = False             # turn off contour lines
res.cnLineLabelsOn         = False             # turn off contour labels
res.cnLevelSpacingF        = 0.5   # 2 was chosen

res.lbOrientation          = "Horizontal"      # vertical by default
res.lbBoxLinesOn           = False             # turn off labelbar boxes
res.lbLabelFontHeightF     = 0.01

res.mpFillOn               = True
res.mpOutlineOn            = True
res.mpLandFillColor        = "tan"
res.mpOceanFillColor       = "transparent"          # No fill
res.mpInlandWaterFillColor = "transparent"
res.mpFillDrawOrder        = "PostDraw"
res.mpGridAndLimbOn        = False

res.mpDataBaseVersion = "MediumRes"                 # Better outlines

res.sfXArray               = lonCell      # where to overlay contours
res.sfYArray               = latCell

res.tiMainString           = "MPAS Ocean grid - raster fill"
res.tiMainOffsetYF         = 0.03      # Move main title towards plot

plot = Ngl.contour_map(wks,temp,res)  

res.cnFillPalette          = "BlAqGrYeOrReVi200"
res.cnLevelSelectionMode   = "ManualLevels"
res.cnMinLevelValF         = 12
res.cnMaxLevelValF         = 27
res.cnLevelSpacingF        = 0.125   # 2 was chosen

res.mpLimitMode            = "LatLon"                    # Zoom in on map
res.mpMinLonF              = -60
res.mpMaxLonF              =   0
res.mpMinLatF              =   0
res.mpMaxLatF              =  40

res.nglDraw               = False
res.nglFrame              = False

plot = Ngl.contour_map(wks,temp,res)  
add_mpas_edges(wks,plot,f)
Ngl.draw(plot)
Ngl.frame(wks)

Ngl.end()

