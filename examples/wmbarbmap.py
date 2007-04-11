#
#  File:
#    wmbarbmap.py
#
#  Synopsis:
#    Draws several wind barbs over a map.
#
#  Category:
#    Wind barbs
#
#  Author:
#    Fred Clare
#  
#  Date of initial publication:
#    March, 2005
#
#  Description:
#    This example draws multiple wind barbs at various lat/lon
#    positions over an orthographic map projection.  It illustrates
#    how the wind barb directions are attenuated to accommodate for
#    the latitude position.
#
#  Effects illustrated:
#    o  Drawing wind barbs over maps.
#    o  A map using an orthographic projection.
#    o  Setting wind barb control parameters.
# 
#  Output:
#    A single visualization is produced showing several wind barbs
#    at different lat/lon positions.
#
#  Notes:
#     

import Ngl 
import numpy

#
#  Draw some wind barbs over a map.
#

#
#  Specify a color map and open an output workstation.
#
cmap = numpy.array([[1., 1., 1.], [0., 0., 0.], [1., 0., 0.]], 'f')
rlist = Ngl.Resources()
rlist.wkColorMap = cmap
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"wmbarbmap",rlist)  # Open a workstation.

#
#  Set some map resources.
#
mpres = Ngl.Resources()
mpres.mpProjection = "Orthographic"
mpres.mpLimitMode  = "LatLon"
mpres.mpMinLonF    = -40.
mpres.mpMaxLonF    =  20.
mpres.mpMinLatF    =  55.
mpres.mpMaxLatF    =  85.
mpres.mpCenterLatF =  70.
mpres.mpCenterLonF = -10.
mpres.nglFrame     = False

#
#  Draw the map.
#
map = Ngl.map(wks,mpres)

#
#  Draw an array of vertical wind barbs over the above map.
#
lat = numpy.zeros([3,2,5],'f')
lon = numpy.zeros([3,2,5],'f')
u   = numpy.zeros([3,2,5],'f')
v   = numpy.zeros([3,2,5],'f')

lat[0,:,:] = 65
lat[1,:,:] = 70
lat[2,:,:] = 75

for i in range(5):
  lon[:,0,i] = -40.+i*5.
  lon[:,1,i] = -15.+i*5.

u[:,:,:] =  0.
v[:,:,:] = 90.

Ngl.wmsetp("col", 2)                 # Draw in red.
Ngl.wmsetp("wbs", .06)               # Increase the size of the barbs.
Ngl.wmbarbmap(wks, lat, lon,  u, v)  # Plot barbs.
Ngl.frame(wks)      

Ngl.end()
