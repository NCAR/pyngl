import os
import Ngl, Nio
from utils import *

dirc = os.path.join("$NCARGTEST","nclscripts","cdf_files")
b    = Nio.open_file(os.path.join(dirc,"wrftest_for.nc"))

tk_for    = b.variables["tk"][:]
rh_for    = b.variables["rh"][:]
tk_m_for  = multid(tk_for,[3,2,2])
rh_m_for  = multid(rh_for,[3,2,2])

filename = "wrfout_d01_2005-12-14_13:00:00.GWATC_FCST"
a = Nio.open_file(os.path.join(dirc,filename+".nc"))

Qv = a.variables["QVAPOR"][:]
P  = a.variables["P"][:]      # perturbation
Pb = a.variables["PB"][:]     # base state pressure"
P  = P + Pb                   # total pressure

theta = a.variables["T"][:]   # perturbation potential temperature (theta-t0)"]
theta = theta + 300.

Pm     = multid(P,[3,2,2])
Qvm    = multid(Qv,[3,2,2])
thetam = multid(theta,[3,2,2])

#---Test the float case
TK_ncl    = Ngl.wrf_tk (P, theta)
RH_ncl    = Ngl.wrf_rh (Qv, P, TK_ncl) 
test_values("wrf_tk",TK_ncl,tk_for)
test_values("wrf_rh",RH_ncl,rh_for)

TK_m_ncl  = Ngl.wrf_tk (Pm, thetam)
RH_m_ncl  = Ngl.wrf_rh (Qvm, Pm, TK_m_ncl)

test_values("wrf_tk",TK_m_ncl,tk_m_for)
test_values("wrf_rh",RH_m_ncl,rh_m_for)

