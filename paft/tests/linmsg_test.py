# To test with the Numeric module, be sure to uncomment these three
# lines, and comment the NumPy lines after this.
#
# import PAF_numeric.Ngl as Ngl
# import MA
# default_type = "numeric"

import sys

# To test with the NumPy module, be sure to uncomment these two lines,
# and comment the three Numeric lines above.
import Ngl
default_type = "numpy"

#
# Leave these both alone, regardless of what module you are testing.
#
import numpy
import Numeric
from utils import *

#
# Begin linmsg tests.
#

#
# Correct value for first set of tests.
#
npts = 101
x     = Ngl.fspan(0.,npts-1,101)
y     = Ngl.fspan(0.,npts-1,101)
xorig = Ngl.fspan(0.,npts-1,101)

xnew = Ngl.linmsg(x)
test_values("linmsg (float)",xnew,xorig)
check_type(xnew,default_type)
del xnew

xnew = Ngl.linmsg(x,0)
test_values("linmsg (float,int)",xnew,xorig)
check_type(xnew,default_type)
del xnew

xnew = Ngl.linmsg(x,0,0)
test_values("linmsg (float,int,int)",xnew,xorig)
check_type(xnew,default_type)
del xnew

fill_value = -999
x[3]  = fill_value
x[7]  = fill_value
x[21] = fill_value
x[50] = fill_value
x[80] = fill_value
x[99] = fill_value

xnew = Ngl.linmsg(x,0,fill_value=fill_value)
test_values("linmsg (some fill_values)",xnew,xorig)
check_type(xnew,default_type)
del xnew

x[0]       = fill_value
x[100]     = fill_value
xorig[0]   = fill_value
xorig[99]  = fill_value
xorig[100] = fill_value

xnew = Ngl.linmsg(x,1,fill_value=fill_value)
test_values("linmsg (end fill_values)",xnew,xorig)
check_type(xnew,default_type)
del xnew

xnew = Ngl.linmsg(x,-1,fill_value=fill_value)
test_values("linmsg (interpolated end fill_values)",xnew[1:99],y[1:99])
test_values("linmsg",[xnew[0],xnew[100]],[xnew[1],xnew[99]])
check_type(xnew,default_type)

del x
del y
del xnew

x = [[[1.,2.,3.],[4.,5.,6.]],[[7.,8.,9.],[10.,11.,12.]]]
y = [[[1.,2.,3.],[4.,5.,6.]],[[7.,8.,9.],[10.,11.,12.]]]

xnew = Ngl.linmsg(x,fill_value=-999)
test_values("linmsg (multi-d, no missing values)",xnew,y)
del xnew

x[1][1][1] = -999
x[1][0][2] = -999
x[0][0][2] = -999
y = [[[1.,2.,2.],[4.,5.,6.] ],[ [7.,8.,8.],[10.,11.,12.]]]
xnew = Ngl.linmsg(x,-1,fill_value=-999)
test_values("linmsg (multi-d, missing values)",xnew,y)


fv = -999
x[0][0][0] = fv
x[1][1][2] = fv
y = [[[fv,2.,fv],[4.,5.,6.] ],[ [7.,8.,fv],[10.,fv,fv]]]
xnew = Ngl.linmsg(x,1,fill_value=fv)
test_values("linmsg (multi-d, missing values)",xnew,y)
test_values("linmsg (multi-d, msg)",[xnew[0][0][0],xnew[0][0][2], \
                                     xnew[1][0][2],xnew[1][1][1], \
                                     xnew[1][1][2]],[fv,fv,fv,fv,fv])
