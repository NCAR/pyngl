import os, sys
import Ngl, Nio
from utils import *

diri     = os.path.join("$NCARGTEST","nclscripts","cdf_files")
filename = os.path.join(diri,"wrfout_d01_2003-07-15_00:00:00")


a_out = Nio.open_file(os.path.join(diri,"wrf_avopvo.nc"))
a     = Nio.open_file(filename + ".nc","r")

avo_out = a_out.variables["avo"][:]
pvo_out = a_out.variables["pvo"][:]

it = 0

u     = a.variables["U"][it,:,:]
v     = a.variables["V"][it,:,:]
theta = a.variables["T"][it,:,:] + 300.
prs   = a.variables["P"][it,:,:] + a.variables["PB"][it,:,:]
msfu  = a.variables["MAPFAC_U"][it,:,:]
msfv  = a.variables["MAPFAC_V"][it,:,:]
msft  = a.variables["MAPFAC_M"][it,:,:]
cor   = a.variables["F"][it,:,:]

dx = 30000.
dy = 30000.

pvo = Ngl.wrf_pvo ( u, v, theta, prs, msfu, msfv, msft, cor, dx, dy, 0)
avo = Ngl.wrf_avo ( u, v, msfu, msfv, msft, cor, dx, dy, 0)

test_values("wrf_avo",avo,avo_out)
test_values("wrf_pvo",pvo,pvo_out,delta=1e-4)

# Test multiple dimensions. msfu, msfv, msft, and cor can be 2D or nD,
# so test both.
#
um       = multid(u,[2,3])
vm       = multid(v,[2,3])
thetam   = multid(theta,[2,3])
prsm     = multid(prs,[2,3])
msfum    = multid(msfu,[2,3])
msfvm    = multid(msfv,[2,3])
msftm    = multid(msft,[2,3])
corm     = multid(cor,[2,3])
pvom_out = multid(pvo,[2,3])
avom_out = multid(avo,[2,3])
pvom = Ngl.wrf_pvo ( um, vm, thetam, prsm, msfu, msfv, msft, cor, dx, dy, 0)
avom = Ngl.wrf_avo ( um, vm, msfu, msfv, msft, cor, dx, dy, 0)

test_values("wrf_pvo",pvom,pvom_out)
test_values("wrf_avo",avom,avom_out)

