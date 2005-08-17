import Ngl,Numeric
from Scientific.IO.NetCDF import NetCDFFile

#
#  Open the netCDF file containing the input data.
#
dirc  = Ngl.ncargpath("data")
cfile = NetCDFFile(dirc+"/cdf/vinth2p.nc","r")

#
#  Define the surface pressure value.
#
p0mb = 1000.

#
#  Define the output pressure levels.
#
pnew = [900.,800.,700.,600.,500.,400.]

#
#  Extract the desired variables.
#
hyam = cfile.variables["hyam"][:]
hybm = cfile.variables["hybm"][:]
T    = (cfile.variables["T"][:,:,:,:])
psrf = (cfile.variables["PS"][:,:,:])

#
#  Do the interpolation.
#
Tnew = Ngl.vinth2p(T,hyam,hybm,pnew,psrf,1,p0mb,1,True)

ntime, output_levels, nlat, nlon = Tnew.shape
print "vinth2p: shape of returned array   = [%1d,%1d,%2d,%3d]" % (Tnew.shape)
print "  number of timesteps     = %4d" % (ntime)
print "  number of output levels = %4d" % (output_levels)
print "  number of latitudes     = %4d" % (nlat)
print "  number of longitudes    = %4d" % (nlon)

#
#  Uncomment the following to print out the resultant array values.
#
# for m in xrange(ntime):
#   for i in xrange(output_levels):
#     for j in xrange(nlat):
#       for k in xrange(nlon):
#         print "%3d%3d%3d%4d%12.5f" % (m,i,j,k,Tnew[m,i,j,k])
