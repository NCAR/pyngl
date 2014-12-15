import os, sys
import Ngl, Nio
from utils import *

dirc    = os.path.join("$NCARGTEST","nclscripts","cdf_files")
zm       = numpy.zeros([2,27,54,86],'f')
tkm      = numpy.zeros([2,27,54,86],'f')
pm       = numpy.zeros([2,27,54,86],'f')
qvm      = numpy.zeros([2,27,54,86],'f')
slpm_out = numpy.zeros([2,54,86],'f')

#
# Do individual tests first, but collect data to do multiple 
# dimension test later.
#
for i in range(2):
    a = Nio.open_file(os.path.join(dirc,"wrf_slp") + str(i) + ".nc")

    z       = a.variables["z"][:]
    tk      = a.variables["tk"][:]
    p       = a.variables["p"][:]
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

    slp_out = a.variables["slp"][:] / 100.

    zm[i,:,:,:]     = z
    tkm[i,:,:,:]    = tk
    pm[i,:,:,:]     = p
    qvm[i,:,:,:]    = qv
    slpm_out[i,:,:] = slp_out

    qv_orig = qv
    slp  = Ngl.wrf_slp(z, tk, p, qv)

    test_values("wrf_slp",qv,qv_orig)

    test_values("wrf_slp",slp,slp_out,delta=1.0)

#
# Multiple dimension test.
#
slpm = Ngl.wrf_slp(zm,tkm,pm,qvm)

test_values("wrf_slp",slpm,slpm_out,delta=1.0)

