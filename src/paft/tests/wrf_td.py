import os, sys
import Ngl, Nio
from utils import *

dirc    = os.path.join("$NCARGTEST","nclscripts","cdf_files")
pm      = numpy.zeros([2,27,54,86],'f')
qvm     = numpy.zeros([2,27,54,86],'f')
tdm_out = numpy.zeros([2,27,54,86],'f')

#
# Do individual tests first, but collect data to do multiple 
# dimension test later.
#
for i in range(2):
    a = Nio.open_file(os.path.join(dirc,"wrf_td") + str(i) + ".nc")
    p      = a.variables["p"][:]
    td_out = a.variables["td"][:]

#
# The below is just a test to make sure that the wrf_slp
# wrapper code for setting qv < 0 to 0 is working. I take
# any values equal to zero, and randomly set them to 
# random negative values.
#
    qv1d = numpy.ravel(a.variables["qv"][:])
    ii   = numpy.equal(qv1d,0.)
    qv1d[ii[::2]]  = -1
    qv1d[ii[1::2]] = -100
    qv1d[ii[2::2]] = -3000.234

    qv = numpy.reshape(qv1d,a.variables["qv"][:].shape)

    pm[i,:,:,:]      = p
    qvm[i,:,:,:]     = qv
    tdm_out[i,:,:,:] = td_out

    td = Ngl.wrf_td( p, qv)

    test_values("wrf_td",td,td_out,delta=1e-4)

#
# Multiple dimension test.
#
tdm = Ngl.wrf_td(pm,qvm)
test_values("wrf_td",tdm,tdm_out,delta=1e-4)

#
# Test with a different file.
#
a = Nio.open_file(os.path.join(dirc,"wrf_td2.nc"))

p      = a.variables["p"][:]
qv     = a.variables["qv"][:]
td_out = a.variables["td"][:]

td = Ngl.wrf_td( p, qv)
test_values("wrf_td",td,td_out,delta=1e-5)
