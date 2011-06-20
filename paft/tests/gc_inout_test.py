import Ngl
default_type = "numpy"

import numpy
from utils import *

plat =  0.0
plon = -2.0
lat = [0.0,  1.0,  0.0, -1.0,  0.0]
lon = [-3.0, -2.0, -1.0, -2.0, -3.0]
inout1 = Ngl.gc_inout(0.0, -2.0, lat,lon)
inout2 = Ngl.gc_inout(0.0,  0.0, lat,lon)
test_value("gc_inout",inout1,True,)
test_value("gc_inout",inout2,False)


lat = [[[  0.0,   0.0,  2.0], [ 0.0,  0.0,  1.0] ],  \
       [[  0.0,   0.0,  2.0], [ 0.0,  0.0,  1.0] ],  \
       [[ 89.0,  89.0, 89.0], [ 0.0,  0.0, 80.0] ]]
lon = [[[ 0.0,    2.0,   1.0], [ -1.0,   1.0, -1.0] ],  \
       [[ 0.0,    2.0,   1.0], [ -1.0,   1.0, -1.0] ],  \
       [[ 0.0,  120.0, 240.0], [  0.0,  90.0, 45.0] ]]

p0_lat = numpy.array([[ 1.0,  0.0],  \
                      [ 1.0, -0.1],  \
                      [90.0, 45.0]])
p0_lon = numpy.array([[ 1.0,  0.0],  \
                      [ 0.0,  0.0],  \
                      [ 0.0, 45.0]])

inout = Ngl.gc_inout(p0_lat, p0_lon, lat, lon)
test_values("gc_inout",inout,[[1,1],[0,0],[1,1]])
