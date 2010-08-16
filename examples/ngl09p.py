#
#  File:
#    ngl09p.py
#
#  Synopsis:
#    Illustrates Animation of contours over a map and using masked arrays.
#
#  Category:
#    Contours over maps
#
#  Author:
#    Fred Clare (based on a code of Mary Haley)
#  
#  Date of initial publication:
#    September, 2005
#
#  Description:
#    This example utilizes masked arrays handle missing data values
#
#  Effects illustrated:
#    o  masked arrays to handle missing values.
#    o  Adding a cyclic point.
#    o  Looping to produce an animation.
#    o  Using a stereographic map projection.
#    o  Many map resource settings via the resource file ngl09p.res.
#    o  Label bar resource settings via the resource file ngl09p.res.
# 
#  Output:
#    The example produces twelve contour plots over a map using
#    a stereographic map projection.
#
#  Notes:
#    This example requires the resource file ngl09p.res.
#     

import numpy
#
#  Import the masked array module.
#
from numpy import ma as MA
import os

#
#  Import Ngl support functions.
#
import Ngl

#
#  Import Nio for a NetCDF reader.
#
import Nio

#
#  To use the ScientificPython module to read in the netCDF file,
#  comment out the above "import" command, and uncomment the 
#  import line below.
#
# from Scientific.IO.NetCDF import NetCDFFile

#
#  Open the netCDF files, get variables.
#
data_dir = Ngl.pynglpath("data")
ice1     = Nio.open_file(os.path.join(data_dir,"cdf","fice.nc"),"r")

#
#  This is the ScientificPython method for opening a netCDF file.
#
# ice1     = NetCDFFile(data_dir + "/cdf/fice.nc","r")

#
#  Create a masked array to accommodate missing values in the fice variable.
#
fice = ice1.variables["fice"]  # fice[120,49,100]
ficea = fice[:,:,:]
fill_value = None
if (hasattr(fice,"missing_value")):
  fill_value = fice.missing_value
elif (hasattr(fice,"_FillValue")):
  fill_value = fice._FillVlaue
fice_masked = MA.transpose(MA.masked_values(ficea,fill_value),(1,2,0))

hlat = ice1.variables["hlat"]  # hlat[49]
hlon = ice1.variables["hlon"]  # hlon[100]


dimf     = fice.shape  # Define an array to hold long-term monthly means.
ntime    = fice.shape[0]
nhlat    = fice.shape[1]
nhlon    = fice.shape[2]

nmo    = 0
month  = nmo+1

icemon = MA.zeros((nhlat,nhlon),dtype=float)
for i in xrange(fice_masked.shape[0]):
  for j in xrange(fice_masked.shape[1]):
    icemon[i,j] = MA.average(fice_masked[i,j,0:ntime:12])

#
#  Fill the places where icemon is zero with the fill value.
#
icemon = MA.masked_values(icemon,0.,rtol=0.,atol=1.e-15)
icemon = MA.filled(icemon,fill_value)

                       # Calculate the January (nmo=0) average.


nsub = 16 # Subscript location of northernmost hlat to be plotted.

cmap = numpy.array([                                         \
         [1.00,1.00,1.00], [0.00,0.00,0.00], [1.00,1.00,0.50], \
         [0.00,0.00,0.50], [0.50,1.00,1.00], [0.50,0.00,0.00], \
         [1.00,0.00,1.00], [0.00,1.00,1.00], [1.00,1.00,0.00], \
         [0.00,0.00,1.00], [0.00,1.00,0.00], [1.00,0.00,0.00], \
         [0.50,0.00,1.00], [1.00,0.50,0.00], [0.00,0.50,1.00], \
         [0.50,1.00,0.00], [0.50,0.00,0.50], [0.50,1.00,0.50], \
         [1.00,0.50,1.00], [0.00,0.50,0.00], [0.50,0.50,1.00], \
         [1.00,0.00,0.50], [0.50,0.50,0.00], [0.00,0.50,0.50], \
         [1.00,0.50,0.50], [0.00,1.00,0.50], [0.50,0.50,0.50], \
         [0.625,0.625,0.625] ],dtype=float)

rlist = Ngl.Resources()
rlist.wkColorMap = cmap
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"ngl09p",rlist) # Open a workstation.

resources = Ngl.Resources()
resources.sfMissingValueV = fill_value

icemonnew,hlonnew = Ngl.add_cyclic(icemon[0:nsub+1,:],hlon[:])
resources.sfXArray = hlonnew   # Necessary for overlay on a map.
resources.sfYArray = hlat[0:nsub+1]
resources.nglSpreadColors = False    # Do not interpolate color space.

resources.tiMainString = "CSM Y00-99 Mean Ice Fraction Month =" + str(month)

resources.pmTickMarkDisplayMode = "Never"

map = Ngl.contour_map(wks,icemonnew,resources) # Draw a contour
                                               # over a map.

nmos = 12    # Specify the number of months in the loop (max 120).
for nmo in range(1,nmos): 
  month  = nmo+1
  for i in xrange(fice_masked.shape[0]):
    for j in xrange(fice_masked.shape[1]):
      icemon[i,j] = MA.average(fice_masked[i,j,nmo:ntime:12])
  icemon = MA.masked_values(icemon,0.,rtol=0.,atol=1.e-15)
  icemon = MA.filled(icemon,fill_value)

  resources.tiMainString = "CSM Y00-99 Mean Ice Fraction Month =" + str(month)
  map = \
    Ngl.contour_map(wks,Ngl.add_cyclic(icemon[0:nsub+1,:]),resources)

del icemon       # Clean up.
del icemonnew 
del map 

Ngl.end()
