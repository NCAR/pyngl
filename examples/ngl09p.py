#
#  Import the Masked array module from 
#  Numerical Python; import sys
#
import Numeric, MA,sys

#
#  Import Ngl support functions.
#
import Ngl

#
#  Import all names from the NetCDF module.
#
from Scientific.IO.NetCDF import NetCDFFile

def addcyclic(data,coord_array):
#
# Add a cyclic point in "x" to a 2D array
# for a lat/lon plot "x"  corresponds to "lon"
#                    "ny" corresponds to "nlat"
#                    "mx" corresponds to "mlon"
#
  if (type(data) != type(Numeric.array([0]))):
    print "addcyclic: input must be a Numeric array."
    sys.exit()

  dims = data.shape
  if (len(dims) != 2):
    print "addcyclic: input must be a 2D Numeric array."
    sys.exit()

  ny   = dims[0]
  mx   = dims[1]
  mx1  = mx+1

  newdata  = Numeric.zeros((ny,mx1),MA.Float0)
  newcoord = Numeric.zeros((mx1),MA.Float0)

  newdata[:,0:mx] = data
  newdata[:,mx]   = data[:,0]
  newcoord[0:mx]  = coord_array
  newcoord[mx]    = coord_array[0]+360.

  return newdata,newcoord

#
#  Open the netCDF files, get variables.
#
data_dir = Ngl.ncargpath("data")
ice1     = NetCDFFile(data_dir + "/cdf/fice.nc","r")

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

icemon = MA.zeros((nhlat,nhlon),MA.Float0)
for i in xrange(fice_masked.shape[0]):
  for j in xrange(fice_masked.shape[1]):
    icemon[i,j] = MA.average(fice_masked[i,j,0:ntime:12])

#
#  Fill the places where icemon is zero with the fill value.
#
icemon = MA.masked_values(icemon,0.,rtol=0.,atol=1.e-15)
icemon = MA.filled(icemon,value=fill_value)

                       # Calculate the January (nmo=0) average.


nsub = 16 # Subscript location of northernmost hlat to be plotted.

cmap = Numeric.array([                                         \
         [1.00,1.00,1.00], [0.00,0.00,0.00], [1.00,1.00,0.50], \
         [0.00,0.00,0.50], [0.50,1.00,1.00], [0.50,0.00,0.00], \
         [1.00,0.00,1.00], [0.00,1.00,1.00], [1.00,1.00,0.00], \
         [0.00,0.00,1.00], [0.00,1.00,0.00], [1.00,0.00,0.00], \
         [0.50,0.00,1.00], [1.00,0.50,0.00], [0.00,0.50,1.00], \
         [0.50,1.00,0.00], [0.50,0.00,0.50], [0.50,1.00,0.50], \
         [1.00,0.50,1.00], [0.00,0.50,0.00], [0.50,0.50,1.00], \
         [1.00,0.00,0.50], [0.50,0.50,0.00], [0.00,0.50,0.50], \
         [1.00,0.50,0.50], [0.00,1.00,0.50], [0.50,0.50,0.50], \
         [0.625,0.625,0.625] ],MA.Float0)

rlist = Ngl.Resources()
rlist.wkColorMap = cmap
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"ngl09p",rlist) # Open a workstation.

resources = Ngl.Resources()
resources.sfMissingValueV = fill_value

icemonnew,hlonnew = addcyclic(icemon[0:nsub+1,:],hlon[:])
resources.sfXArray = hlonnew   # Necessary for overlay on a map.
resources.sfYArray = hlat[0:nsub+1]
resources.nglSpreadColors = False    # Do not interpolate color space.

resources.tiMainString = "CSM Y00-99 Mean Ice Fraction Month =" + str(month)

map = Ngl.contour_map(wks,icemonnew,resources) # Draw a contour
                                               # over a map.

nmos = 12    # Specify the number of months in the loop (max 120).
for nmo in range(1,nmos): 
  month  = nmo+1
  for i in xrange(fice_masked.shape[0]):
    for j in xrange(fice_masked.shape[1]):
      icemon[i,j] = MA.average(fice_masked[i,j,nmo:ntime:12])
  icemon = MA.masked_values(icemon,0.,rtol=0.,atol=1.e-15)
  icemon = MA.filled(icemon,value=fill_value)

  resources.tiMainString = "CSM Y00-99 Mean Ice Fraction Month =" + str(month)
  map = \
    Ngl.contour_map(wks,(addcyclic(icemon[0:nsub+1,:],hlon[:]))[0],resources)

del icemon       # Clean up.
del icemonnew 
del map 

Ngl.end()
