#
#  File:
#    vinth2p.py
#
#  Synopsis:
#    Illustrates a call to the vinth2p function.
#
#  Category:
#    Processing.
#
#  Author:
#    Fred Clare
#  
#  Date of initial publication:
#    August, 2005
#
#  Description:
#    Reads from a NetCDF file, creates arguments and calls
#    vinth2p, prints some return values.
#
#  Effects illustrated:
#    o  Reading from a NetCDF file.
#    o  A call to vinth2p.
# 
#  Output:
#    Some of the returned variables are printed to standard 
#    output.  No visualization is produced.
#
#  Notes:
#     

import Ngl
import numpy
import Nio
import os

#
#  Open the netCDF file containing the input data.
#
dirc  = Ngl.pynglpath("data")
cfile = Nio.open_file(os.path.join(dirc,"cdf","vinth2p.nc"),"r")

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
print "  number of input levels  = %4d" % (T.shape[1])
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
