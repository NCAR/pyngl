#
#  File:
#    mpas3.py
#
#  Synopsis:
#    Draws outlines of an MPAS grid with 349,339 edges
#
#  Categories:
#    polylines
#
#  Author:
#    Mary Haley, based on an NCL example by Dave Brown
#  
#  Date of initial publication:
#    September 2014
#
#  Description:
#    This example shows how to draw a subset of the outline of an MPAS grid.
#
#  Effects illustrated:
#    o  Plotting edges of MPAS data on a map
#    o  Using special "gsSegments" for faster polyline draw
#    o  Zooming in on a map
# 
#  Output:
#    This example produces one plot.
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

mpas_file = "MPASOcean60km.nc"

#---Read edge and lat/lon information
f              = Nio.open_file(mpas_file)
lonCell        = f.variables["lonCell"][:]
latCell        = f.variables["latCell"][:]
lonVertex      = f.variables["lonVertex"][:]
latVertex      = f.variables["latVertex"][:]
verticesOnEdge = f.variables["verticesOnEdge"][:]
nedges         = verticesOnEdge.shape[0]

#---Convert to degrees from radians
RAD2DEG   = 180.0/(math.atan(1)*4.0)
latCell   = latCell * RAD2DEG
lonCell   = lonCell * RAD2DEG
latVertex = latVertex * RAD2DEG
lonVertex = lonVertex * RAD2DEG

#---Start the graphics
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"mpas3")

res                   = Ngl.Resources()             # Plot mods desired.

res.nglDraw           = False                       # Draw plot later after
res.nglFrame          = False                       # adding edge outlines.

res.mpProjection      = "CylindricalEquidistant"    # The default
res.mpOutlineOn       = True
res.mpDataBaseVersion = "MediumRes"                 # Better outlines

res.mpLimitMode       = "LatLon"                    # Zoom in on map
res.mpMinLonF         = -60
res.mpMaxLonF         =   0
res.mpMinLatF         =   0
res.mpMaxLatF         =  40

res.mpFillOn          = True
res.mpLandFillColor   = "tan"
res.mpOceanFillColor  = "transparent"               # No fill
res.mpGridAndLimbOn   = False

res.tiMainString      = mpas_file

map = Ngl.map(wks,res)  # Create the map, don't draw it.

#---Code to attach MPAS edge lines to the existing map
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

poly = Ngl.add_polyline(wks,map,numpy.ravel(ecx),numpy.ravel(ecy),lnres)

Ngl.draw(map)
Ngl.frame(wks)

Ngl.end()

